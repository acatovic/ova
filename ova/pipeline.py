from enum import Enum
import io
import os
import re
import signal
import wave

# Patch missing SIGKILL on Windows so nemo-toolkit can import
if os.name == "nt" and not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM  # type: ignore[attr-defined]

from kokoro import KPipeline
import nemo.collections.asr as nemo_asr
import numpy as np
from ollama import chat as ollama_chat
from pocket_tts import TTSModel as PocketTTSModel
import requests
import torch

from .audio import numpy_to_wav_bytes, resample
from .utils import get_device, logger


DEFAULT_SR = 24000  # default sample rate
DEFAULT_TTS_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_TTS_VOICE = "af_heart"
DEFAULT_CHAT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
DEFAULT_ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

POCKET_TTS_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
DEFAULT_POCKET_TTS_VOICE = "alba"

DEFAULT_KOBOLDCPP_URL = "http://localhost:5001"


class OVAProfile(str, Enum):
    DEFAULT = "default"
    DUA = "dua"


class TTSEngine(str, Enum):
    KOKORO = "kokoro"
    POCKET_TTS = "pocket_tts"


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    KOBOLDCPP = "koboldcpp"


class OVAPipeline:
    def __init__(
        self,
        profile: OVAProfile | str,
        tts_engine: TTSEngine | str = TTSEngine.KOKORO,
        llm_backend: LLMBackend | str = LLMBackend.OLLAMA,
        llm_model: str | None = None,
        koboldcpp_url: str = DEFAULT_KOBOLDCPP_URL,
        pocket_tts_voice: str = DEFAULT_POCKET_TTS_VOICE,
    ):
        try:
            self.profile = OVAProfile(profile)
        except ValueError:
            logger.warning(f"Unknown OVA profile '{profile}', defaulting to DEFAULT")
            self.profile = OVAProfile.DEFAULT

        try:
            self._tts_engine = TTSEngine(tts_engine)
        except ValueError:
            logger.warning(f"Unknown TTS engine '{tts_engine}', defaulting to KOKORO")
            self._tts_engine = TTSEngine.KOKORO

        try:
            self._llm_backend = LLMBackend(llm_backend)
        except ValueError:
            logger.warning(f"Unknown LLM backend '{llm_backend}', defaulting to OLLAMA")
            self._llm_backend = LLMBackend.OLLAMA

        self.device = get_device()

        # LLM configuration
        self.chat_model = llm_model or DEFAULT_CHAT_MODEL
        self.koboldcpp_url = koboldcpp_url.rstrip("/")

        # Pocket-TTS voice
        self._pocket_tts_voice = pocket_tts_voice if pocket_tts_voice in POCKET_TTS_VOICES else DEFAULT_POCKET_TTS_VOICE

        # prep for loading assistant profile / prompt
        profile_dir = f"profiles/{self.profile.value}"

        with open(f"{profile_dir}/prompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

        self.context = [{"role": "system", "content": self.system_prompt}]

        # initialize TTS models (lazy — only load the active engine)
        self._kokoro_model = None
        self._pocket_tts_model = None
        self._pocket_tts_voice_states = {}

        self._init_tts_engine(profile_dir)

        # initialize ASR
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=DEFAULT_ASR_MODEL)

    # ------------------------------------------------------------------
    # TTS engine management
    # ------------------------------------------------------------------

    def _init_tts_engine(self, profile_dir: str | None = None):
        """Load the currently selected TTS engine's model."""
        if self._tts_engine == TTSEngine.KOKORO:
            self._ensure_kokoro()
        elif self._tts_engine == TTSEngine.POCKET_TTS:
            self._ensure_pocket_tts()

    def _ensure_kokoro(self):
        if self._kokoro_model is None:
            logger.info("Loading Kokoro TTS model...")
            self._kokoro_model = KPipeline(lang_code='a', repo_id=DEFAULT_TTS_MODEL)
            # warm up
            self._kokoro_model("Just testing!", voice=DEFAULT_TTS_VOICE)
            logger.info("Kokoro TTS model loaded.")

    def _ensure_pocket_tts(self):
        if self._pocket_tts_model is None:
            logger.info("Loading Pocket-TTS model...")
            self._pocket_tts_model = PocketTTSModel.load_model()
            logger.info("Pocket-TTS model loaded.")
        # Pre-load current voice state if not cached
        self._get_pocket_tts_voice_state(self._pocket_tts_voice)

    def _get_pocket_tts_voice_state(self, voice: str):
        """Get or cache a Pocket-TTS voice state."""
        if voice not in self._pocket_tts_voice_states:
            logger.info(f"Loading Pocket-TTS voice: {voice}")
            self._pocket_tts_voice_states[voice] = self._pocket_tts_model.get_state_for_audio_prompt(voice)
        return self._pocket_tts_voice_states[voice]

    @property
    def tts_engine(self) -> TTSEngine:
        return self._tts_engine

    @property
    def llm_backend(self) -> LLMBackend:
        return self._llm_backend

    @property
    def pocket_tts_voice(self) -> str:
        return self._pocket_tts_voice

    def set_tts_engine(self, engine: TTSEngine | str):
        """Switch the active TTS engine at runtime."""
        try:
            engine = TTSEngine(engine)
        except ValueError:
            raise ValueError(f"Unknown TTS engine: {engine}")

        self._tts_engine = engine
        profile_dir = f"profiles/{self.profile.value}"
        self._init_tts_engine(profile_dir)
        logger.info(f"TTS engine switched to: {engine.value}")

    def set_pocket_tts_voice(self, voice: str):
        """Change the Pocket-TTS voice."""
        if voice not in POCKET_TTS_VOICES:
            raise ValueError(f"Unknown Pocket-TTS voice: {voice}. Available: {POCKET_TTS_VOICES}")
        self._pocket_tts_voice = voice
        if self._pocket_tts_model is not None:
            self._get_pocket_tts_voice_state(voice)
        logger.info(f"Pocket-TTS voice changed to: {voice}")

    def set_llm_backend(self, backend: LLMBackend | str, model: str | None = None, koboldcpp_url: str | None = None):
        """Switch the LLM backend at runtime."""
        try:
            backend = LLMBackend(backend)
        except ValueError:
            raise ValueError(f"Unknown LLM backend: {backend}")

        self._llm_backend = backend
        if model is not None:
            self.chat_model = model
        if koboldcpp_url is not None:
            self.koboldcpp_url = koboldcpp_url.rstrip("/")
        logger.info(f"LLM backend switched to: {backend.value} (model: {self.chat_model})")

    def set_system_prompt(self, prompt: str):
        """Replace the system prompt and reset conversation context."""
        self.system_prompt = prompt
        self.context = [{"role": "system", "content": self.system_prompt}]
        logger.info(f"System prompt updated ({len(prompt)} chars). Conversation context reset.")

    # ------------------------------------------------------------------
    # Text chunking for TTS
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences for TTS processing.

        TTS engines can fail on long input. This splits on sentence
        boundaries so each chunk is a manageable size.
        """
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty strings
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # TTS methods
    # ------------------------------------------------------------------

    def tts(self, text: str) -> bytes:
        """Route to the active TTS engine."""
        if self._tts_engine == TTSEngine.KOKORO:
            return self._tts_kokoro(text)
        elif self._tts_engine == TTSEngine.POCKET_TTS:
            return self._tts_pocket(text)
        else:
            raise ValueError(f"No TTS handler for engine: {self._tts_engine}")

    def _tts_kokoro(self, text: str) -> bytes:
        self._ensure_kokoro()
        generator = self._kokoro_model(text, voice=DEFAULT_TTS_VOICE)

        chunks = []
        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size > 0:
                chunks.append(audio)

        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return numpy_to_wav_bytes(arr, sr=DEFAULT_SR)

    def _tts_pocket(self, text: str) -> bytes:
        self._ensure_pocket_tts()
        voice_state = self._get_pocket_tts_voice_state(self._pocket_tts_voice)
        sr = self._pocket_tts_model.sample_rate

        sentences = self._split_into_sentences(text)
        chunks = []
        for sentence in sentences:
            audio_tensor = self._pocket_tts_model.generate_audio(voice_state, sentence)
            arr = audio_tensor.numpy()
            if arr.size > 0:
                chunks.append(arr)

        combined = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return numpy_to_wav_bytes(combined, sr=sr)

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes or len(wav_bytes) < 44:
            logger.warning("Received empty or too-short WAV data for transcription")
            return ""

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            num_frames = wf.getnframes()
            pcm = wf.readframes(num_frames)

        if num_frames == 0 or len(pcm) == 0:
            logger.warning("WAV file has 0 frames")
            return ""

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

        if audio.size == 0:
            logger.warning("Audio is empty after resampling")
            return ""

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

    # ------------------------------------------------------------------
    # LLM chat
    # ------------------------------------------------------------------

    def chat(self, text: str) -> str:
        """Route to the active LLM backend."""
        if self._llm_backend == LLMBackend.OLLAMA:
            return self._chat_ollama(text)
        elif self._llm_backend == LLMBackend.KOBOLDCPP:
            return self._chat_koboldcpp(text)
        else:
            raise ValueError(f"No chat handler for backend: {self._llm_backend}")

    def _strip_markdown(self, text: str) -> str:
        return text.replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()

    def _chat_ollama(self, text: str) -> str:
        self.context.append({"role": "user", "content": text})

        response = ollama_chat(
            model=self.chat_model,
            messages=self.context,
            think=False,
            stream=False,
        )

        response_text = self._strip_markdown(response.message.content)
        self.context.append({"role": "assistant", "content": response_text})
        return response_text

    def _chat_koboldcpp(self, text: str) -> str:
        self.context.append({"role": "user", "content": text})

        url = f"{self.koboldcpp_url}/v1/chat/completions"
        payload = {
            "model": self.chat_model,
            "messages": self.context,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            response_text = self._strip_markdown(data["choices"][0]["message"]["content"])
        except requests.RequestException as e:
            logger.error(f"Koboldcpp request failed: {e}")
            # Remove the user message we just added since we failed
            self.context.pop()
            raise RuntimeError(f"Koboldcpp request failed: {e}") from e

        self.context.append({"role": "assistant", "content": response_text})
        return response_text

    # ------------------------------------------------------------------
    # Configuration snapshot (for the API)
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            "profile": self.profile.value,
            "tts_engine": self._tts_engine.value,
            "pocket_tts_voice": self._pocket_tts_voice,
            "pocket_tts_voices": POCKET_TTS_VOICES,
            "llm_backend": self._llm_backend.value,
            "llm_model": self.chat_model,
            "koboldcpp_url": self.koboldcpp_url,
            "system_prompt": self.system_prompt,
        }
