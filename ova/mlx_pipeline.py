import os
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.stt.utils import load_model as load_asr_model
from mlx_audio.tts.generate import load_audio as load_tts_ref_audio
from mlx_audio.tts.utils import load_model as load_tts_model
from ollama import chat

from .mlx_audio import mx_to_wav_bytes
from .utils import logger

DEFAULT_SR = 24000  # default sample rate
# VOICE_CLONE_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
# VOICE_CLONE_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
VOICE_CLONE_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
DEFAULT_CHAT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class OVAPipeline:
    def __init__(self, profile: str = "sydney"):
        required_files = ["prompt.txt", "ref_audio.wav", "ref_text.txt"]
        repo_root = Path(__file__).resolve().parent.parent
        profile_dir = repo_root / "profiles" / profile

        missing = [
            filename
            for filename in required_files
            if not (profile_dir / filename).is_file()
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(
                "Unable to load voice profile "
                f"'{profile}' (missing: {missing_text}) at {profile_dir}"
            )

        self.profile = profile
        self.ref_text = (
            (profile_dir / "ref_text.txt").read_text(encoding="utf-8").strip()
        )
        self.ref_audio = load_tts_ref_audio(
            str(profile_dir / "ref_audio.wav"), sample_rate=DEFAULT_SR
        )

        self.system_prompt = (
            (profile_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        )
        self.context = [{"role": "system", "content": self.system_prompt}]

        self.tts_model = load_tts_model(VOICE_CLONE_TTS_MODEL)

        # warm up TTS
        self.tts_model.generate(
            "How'dy folks, we are just taking our TTS for a warmup!",
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
        if not wav_bytes:
            return ""

        with tempfile.TemporaryDirectory(prefix="ova_transcribe_") as tmp_dir:
            audio_path = os.path.join(tmp_dir, "audio.wav")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(wav_bytes)

            transcript_path = os.path.join(tmp_dir, "transcript")
            result = generate_transcription(
                model=self.asr_model,
                audio=audio_path,
                output_path=transcript_path,
                format="txt",
            )

        if hasattr(result, "text"):
            return result.text.strip()
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()

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
