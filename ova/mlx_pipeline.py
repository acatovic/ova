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

DEFAULT_SR = 24000  # default sample rate
VOICE_CLONE_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
DEFAULT_CHAT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
DEFAULT_ASR_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class OVAPipeline:
    def __init__(self, profile: str = "sydney"):
        repo_root = Path(__file__).resolve().parent.parent
        profile_dir = repo_root / "profiles" / profile
        prompt_path = profile_dir / "prompt.txt"
        ref_audio_path = self._find_reference_audio(profile_dir)

        if not prompt_path.is_file() or ref_audio_path is None:
            raise RuntimeError(
                "Unable to load voice profile "
                f"'{profile}' at {profile_dir}. Expected prompt.txt and at least one .wav file under audio/"
            )

        self.profile = profile

        self.system_prompt = (
            (profile_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        )
        self.context = [{"role": "system", "content": self.system_prompt}]

        # initialize ASR
        self.asr_model = load_asr_model(DEFAULT_ASR_MODEL)
        self.ref_text = self._transcribe_reference_audio(ref_audio_path)
        self.ref_audio = load_tts_ref_audio(str(ref_audio_path), sample_rate=DEFAULT_SR)

        self.tts_model = load_tts_model(VOICE_CLONE_TTS_MODEL)

        # warm up TTS
        self.tts_model.generate(
            "How'dy folks, we are just taking our TTS for a warmup!",
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
        )

        # initialize chat model
        self.chat_model = DEFAULT_CHAT_MODEL

    def _find_reference_audio(self, profile_dir: Path) -> Path | None:
        audio_dir = profile_dir / "audio"
        if not audio_dir.is_dir():
            return None

        audio_files = sorted(
            path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
        )
        if not audio_files:
            return None

        return audio_files[0]

    def _transcribe_reference_audio(self, ref_audio_path: Path) -> str:
        with tempfile.TemporaryDirectory(prefix="ova_profile_transcribe_") as tmp_dir:
            transcript_path = os.path.join(tmp_dir, "transcript")
            result = generate_transcription(
                model=self.asr_model,
                audio=str(ref_audio_path),
                output_path=transcript_path,
                format="txt",
            )

        if hasattr(result, "text"):
            ref_text = result.text.strip()
        elif isinstance(result, str):
            ref_text = result.strip()
        else:
            ref_text = str(result).strip()

        if not ref_text:
            raise RuntimeError(
                f"Unable to extract reference text from {ref_audio_path} for profile '{self.profile}'."
            )

        return ref_text

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
            .replace("#", "")
            .strip()
        )

        self.context.append({"role": "assistant", "content": response})

        return response
