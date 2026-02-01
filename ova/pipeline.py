from enum import Enum
import io
from pathlib import Path
import wave

from kokoro import KPipeline
import nemo.collections.asr as nemo_asr
import numpy as np
from ollama import chat
from qwen_tts import Qwen3TTSModel
import torch

from .audio import numpy_to_wav_bytes, resample
from .utils import get_device, logger


DEFAULT_SR = 24000  # default sample rate
DEFAULT_TTS_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_TTS_VOICE = "af_heart"
VOICE_CLONE_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_CHAT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
DEFAULT_ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class OVAPipeline:
    def __init__(self, profile: str = "default"):
        required_files = ["prompt.txt", "ref_audio.wav", "ref_text.txt"]
        profile_dir = Path(f"profiles/{profile}/")

        if profile_dir.is_dir() and all((profile_dir / f).is_file() for f in required_files):
            self.profile = profile
            self.tts = self._tts_with_voice_clone
        else:
            self.profile = "default"
            self.tts = self._tts
            if profile != "default":
                logger.warning((
                    f"Unknown OVA profile '{profile}' or missing the following files in 'profiles/{profile}/' directory: "
                    f"{', '.join(required_files)}. Using 'default' profile."
                ))

        self.device = get_device()

        with open(profile_dir / "prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

        self.context = [{"role": "system", "content": self.system_prompt}]

        # initialize TTS
        if self.tts.__name__ == "_tts_with_voice_clone":  # voice cloning
            self.tts_model = Qwen3TTSModel.from_pretrained(
                VOICE_CLONE_TTS_MODEL,
                device_map=self.device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

            with open(profile_dir / "ref_text.txt", "r", encoding="utf-8") as f:
                ref_text = f.read().strip()

            self.voice_clone_prompt_items = self.tts_model.create_voice_clone_prompt(
                ref_audio=str(profile_dir / "ref_audio.wav"),
                ref_text=ref_text,
                x_vector_only_mode=False,
            )
        else:  # default / super-fast TTS
            self.tts_model = KPipeline(lang_code='a', repo_id=DEFAULT_TTS_MODEL)  # 'a' => US/American English

            # warm up
            self.tts_model("Just testing!", voice=DEFAULT_TTS_VOICE)
        
        # initialize ASR
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=DEFAULT_ASR_MODEL)

        # initialize chat model
        self.chat_model = DEFAULT_CHAT_MODEL


    def _tts(self, text: str) -> bytes:
        generator = self.tts_model(text, voice=DEFAULT_TTS_VOICE)

        chunks = []
        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size > 0:
                chunks.append(audio)

        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        wav_bytes = numpy_to_wav_bytes(arr, sr=DEFAULT_SR)

        return wav_bytes


    def _tts_with_voice_clone(self, text: str) -> bytes:
        wavs, sr = self.tts_model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=self.voice_clone_prompt_items,
        )

        wav_bytes = numpy_to_wav_bytes(wavs[0], sr)

        return wav_bytes


    def transcribe(self, wav_bytes: bytes) -> str:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            num_frames = wf.getnframes()
            pcm = wf.readframes(num_frames)

        # PCM -> float32 in [-1, 1]
        if sampwidth == 1:
            audio = np.frombuffer(pcm, dtype=np.uint8).astype(np.int16) - 128
            audio = audio.astype(np.float32) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            a_f32 = np.frombuffer(pcm, dtype=np.float32)
            if np.isfinite(a_f32).all() and (np.abs(a_f32).max() <= 10.0) and (np.abs(a_f32).mean() < 0.5):
                audio = a_f32.astype(np.float32)
            else:
                audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        # Downmix to mono if needed
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1).astype(np.float32)

        # Resample to model SR
        model_sr = int(getattr(getattr(self.asr_model, "cfg", None), "sample_rate", DEFAULT_SR))
        audio = resample(audio, src_sr, model_sr)

        # Torch tensors on model device
        device = next(self.asr_model.parameters()).device
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, T]
        length_tensor = torch.tensor([audio.shape[0]], device=device, dtype=torch.long)

        self.asr_model.eval()
        with torch.inference_mode():
            out = self.asr_model(input_signal=audio_tensor, input_signal_length=length_tensor)

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                logits, logit_lengths = out[0], out[1]
            elif isinstance(out, dict):
                logits = out.get("logits", out.get("encoded"))
                logit_lengths = out.get("logit_lengths", out.get("encoded_len"))
                if logits is None or logit_lengths is None:
                    raise RuntimeError(f"Unexpected model output keys: {list(out.keys())}")
            else:
                raise RuntimeError(f"Unexpected model output type: {type(out)}")

            decoding = getattr(self.asr_model, "decoding", None)
            if decoding is None:
                raise RuntimeError("Model has no `decoding`; cannot decode.")

            if hasattr(decoding, "ctc_decoder_predictions_tensor"):
                texts = decoding.ctc_decoder_predictions_tensor(logits, logit_lengths)
            elif hasattr(decoding, "rnnt_decoder_predictions_tensor"):
                texts = decoding.rnnt_decoder_predictions_tensor(logits, logit_lengths)
            else:
                raise RuntimeError("No supported decoder method found on `asr_model.decoding`.")

        # Extract text from Hypothesis object if needed
        if texts and len(texts) > 0:
            text = texts[0]
            if hasattr(text, 'text'):
                return text.text.strip()
            elif isinstance(text, str):
                return text.strip()
            else:
                return str(text).strip()

        return ""


    def chat(self, text: str) -> str:
        self.context.append({"role": "user", "content": text})

        response = chat(
            model=self.chat_model,
            messages=self.context,
            think=False,
            stream=False,
        )

        response = response.message.content.replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()

        self.context.append({"role": "assistant", "content": response})

        return response
