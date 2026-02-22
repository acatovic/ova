import os
import shutil
import subprocess
import sys
import tempfile
import time

from ova.pipeline import OVAPipeline

OVA_PROFILE = os.getenv("OVA_PROFILE", "default")


def find_audio_player() -> list[str] | None:
    if sys.platform == "darwin":
        candidates = [
            ["afplay"],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"],
            ["mpv", "--no-video", "--really-quiet"],
        ]
    else:
        candidates = [
            ["aplay", "-q"],
            ["paplay"],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"],
            ["mpv", "--no-video", "--really-quiet"],
        ]

    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd

    return None


def play_wav_bytes(wav_bytes: bytes, player_cmd: list[str] | None) -> None:
    if sys.platform == "win32":
        import winsound  # type: ignore[import-not-found]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            path = tmp.name
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME)
        finally:
            os.unlink(path)
        return

    if not player_cmd:
        print(
            "No audio player found. Install one of: aplay, paplay, ffplay, mpv, afplay."
        )
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        path = tmp.name

    try:
        subprocess.run(player_cmd + [path], check=False)
    finally:
        os.unlink(path)


def main():
    pipeline = OVAPipeline(profile=OVA_PROFILE)
    player_cmd = find_audio_player()

    while True:
        text = input("> ").strip()
        if text and not text.startswith("/"):
            audio_out = pipeline.tts(text)
            play_wav_bytes(audio_out, player_cmd)
        else:
            if text == "/exit":
                print("Bye!")
                time.sleep(2)
                sys.exit(0)


if __name__ == "__main__":
    main()
