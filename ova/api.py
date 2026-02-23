import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .pipeline import OVAPipeline

with open(".config") as f:
    backend = f.read().strip().split("=")[1]
    print(f"### BACKEND: {backend}")

OVA_PROFILE = os.getenv("OVA_PROFILE", "default")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = OVAPipeline(profile=OVA_PROFILE)


@app.post("/chat", response_class=Response)
async def chat_request_handler(request: Request):
    audio_in = await request.body()

    transcribed_text = pipeline.transcribe(audio_in)

    if not transcribed_text:
        # return "empty" bytes if no transcription
        return Response(content=bytes(), media_type="audio/wav")

    chat_response = pipeline.chat(transcribed_text)

    audio_out = pipeline.tts(chat_response)

    return Response(content=audio_out, media_type="audio/wav")
