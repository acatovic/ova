import io
import wave
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.utils import load_model as load_asr_model
from mlx_audio.tts.generate import load_audio as load_tts_ref_audio
from mlx_audio.tts.utils import load_model as load_tts_model
from ollama import chat

from .mlx_audio import mx_to_wav_bytes
from .utils import logger

DEFAULT_SR = 24000  # default sample rate
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-4bit"
DEFAULT_TTS_VOICE = "af_heart"
VOICE_CLONE_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
DEFAULT_CHAT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class OVAPipeline:
    def __init__(self, profile: str = "default"):
        required_files = ["prompt.txt", "ref_audio.wav", "ref_text.txt"]
        profile_dir = Path(f"profiles/{profile}/")

        if profile_dir.is_dir() and all(
            (profile_dir / f).is_file() for f in required_files
        ):
            self.profile = profile
            self.ref_audio = None
            self.ref_text = (profile_dir / "ref_text.txt").read_text(
                encoding="utf-8"
            ).strip()

            try:
                self.ref_audio = load_tts_ref_audio(
                    str(profile_dir / "ref_audio.wav"), sample_rate=DEFAULT_SR
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load reference audio for profile '%s': %s",
                    profile,
                    exc,
                )

            try:
                self.tts_model = load_tts_model(VOICE_CLONE_TTS_MODEL)
                if self.ref_audio is None or not self.ref_text:
                    raise RuntimeError("Missing reference audio/text for voice cloning")
            except Exception as exc:
                logger.warning(
                    "Failed to load voice-clone TTS model '%s': %s. Falling back to '%s'.",
                    VOICE_CLONE_TTS_MODEL,
                    exc,
                    DEFAULT_TTS_MODEL,
                )
                self.tts_model = load_tts_model(DEFAULT_TTS_MODEL)
                self.ref_audio = None
                self.ref_text = None
        else:
            self.profile = "default"
            self.tts_model = load_tts_model(DEFAULT_TTS_MODEL)
            self.ref_audio = None
            self.ref_text = None
            if profile != "default":
                logger.warning(
                    (
                        f"Unknown OVA profile '{profile}' or missing the following files in 'profiles/{profile}/' directory: "
                        f"{', '.join(required_files)}. Using 'default' profile."
                    )
                )

        self.context = [
            {"role": "system", "content": "You are a helpful voice assistant"}
        ]

        # warm up TTS
        self.tts_model.generate(
            "Just testing",
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
        )

        # initialize ASR
        self.asr_model = load_asr_model(DEFAULT_ASR_MODEL)

        # initialize chat model
        self.chat_model = DEFAULT_CHAT_MODEL

    def tts(self, text: str) -> bytes:
        results = list(
            self.tts_model.generate(
                text=text,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
            )
        )

        segments = [r.audio for r in results]

        audio = mx.concatenate(segments, axis=0)

        wav_bytes = mx_to_wav_bytes(audio, sr=DEFAULT_SR)

        return wav_bytes

    def transcribe(self, wav_bytes: bytes) -> str:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            num_frames = wf.getnframes()
            pcm = wf.readframes(num_frames)
        pass

    def chat(self, text: str) -> str:
        self.context.append({"role": "user", "content": text})

        response = chat(
            model=self.chat_model,
            messages=self.context,
            think=False,
            stream=False,
        )

        response = (
            response.message.content.replace("**", "")
            .replace("_", "")
            .replace("__", "")
            .replace("#", "")
            .strip()
        )

        self.context.append({"role": "assistant", "content": response})

        return response
