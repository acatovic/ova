import base64
import json
import os
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

with open(".config") as f:
    backend = f.read().strip().split("=")[1]

DEFAULT_OVA_PROFILE = "default"

if backend == "cuda":
    from .pipeline import OVAPipeline
else:
    # mlx
    from .mlx_pipeline import OVAPipeline

    DEFAULT_OVA_PROFILE = "sydney"

OVA_PROFILE = os.getenv("OVA_PROFILE", DEFAULT_OVA_PROFILE)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = OVAPipeline(profile=OVA_PROFILE)


def split_sentences(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


@app.post("/chat", response_class=Response)
async def chat_request_handler(request: Request):
    audio_in = await request.body()

    transcribed_text = pipeline.transcribe(audio_in)

    if not transcribed_text:
        return Response(content=b"", media_type="application/x-ndjson")

    chat_response = pipeline.chat(transcribed_text)

    sentences = split_sentences(chat_response)
    if not sentences:
        sentences = [chat_response.strip()]

    def iter_audio_chunks():
        for idx, sentence in enumerate(sentences):
            wav_bytes = pipeline.tts(sentence)
            payload = {
                "index": idx,
                "text": sentence,
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
            }
            yield (json.dumps(payload) + "\n").encode("utf-8")

    return StreamingResponse(iter_audio_chunks(), media_type="application/x-ndjson")
