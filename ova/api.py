import os
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipeline import OVAPipeline, POCKET_TTS_VOICES
from .utils import logger


OVA_PROFILE = os.getenv("OVA_PROFILE", "default")
OVA_TTS_ENGINE = os.getenv("OVA_TTS_ENGINE", "kokoro")
OVA_LLM_BACKEND = os.getenv("OVA_LLM_BACKEND", "ollama")
OVA_LLM_MODEL = os.getenv("OVA_LLM_MODEL", "")
OVA_KOBOLDCPP_URL = os.getenv("OVA_KOBOLDCPP_URL", "http://localhost:5001")
OVA_POCKET_TTS_VOICE = os.getenv("OVA_POCKET_TTS_VOICE", "alba")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = OVAPipeline(
    profile=OVA_PROFILE,
    tts_engine=OVA_TTS_ENGINE,
    llm_backend=OVA_LLM_BACKEND,
    llm_model=OVA_LLM_MODEL or None,
    koboldcpp_url=OVA_KOBOLDCPP_URL,
    pocket_tts_voice=OVA_POCKET_TTS_VOICE,
)


# ---------- Audio chat (existing endpoint) ----------

@app.post("/chat", response_class=Response)
async def chat_request_handler(request: Request):
    audio_in = await request.body()

    if not audio_in or len(audio_in) < 44:
        return Response(content=bytes(), media_type="audio/wav")

    try:
        transcribed_text = pipeline.transcribe(audio_in)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return Response(content=bytes(), media_type="audio/wav")

    if not transcribed_text:
        return Response(content=bytes(), media_type="audio/wav")

    try:
        chat_response = pipeline.chat(transcribed_text)
        audio_out = pipeline.tts(chat_response)
    except Exception as e:
        logger.error(f"Chat/TTS failed: {e}")
        return Response(content=bytes(), media_type="audio/wav")

    return Response(content=audio_out, media_type="audio/wav")


# ---------- Text chat (new endpoint) ----------

class TextChatRequest(BaseModel):
    text: str


@app.post("/chat/text", response_class=Response)
async def text_chat_handler(request: TextChatRequest):
    if not request.text.strip():
        return Response(content=bytes(), media_type="audio/wav")

    try:
        chat_response = pipeline.chat(request.text.strip())
        audio_out = pipeline.tts(chat_response)
    except Exception as e:
        logger.error(f"Chat/TTS failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    return Response(
        content=audio_out,
        media_type="audio/wav",
        headers={"X-Chat-Response": quote(chat_response, safe="")},
    )


# ---------- Configuration endpoints ----------

@app.get("/config")
async def get_config():
    return JSONResponse(content=pipeline.get_config())


class ConfigUpdate(BaseModel):
    tts_engine: str | None = None
    pocket_tts_voice: str | None = None
    llm_backend: str | None = None
    llm_model: str | None = None
    koboldcpp_url: str | None = None
    system_prompt: str | None = None


@app.post("/config")
async def update_config(update: ConfigUpdate):
    errors = []

    if update.tts_engine is not None:
        try:
            pipeline.set_tts_engine(update.tts_engine)
        except ValueError as e:
            errors.append(str(e))

    if update.pocket_tts_voice is not None:
        try:
            pipeline.set_pocket_tts_voice(update.pocket_tts_voice)
        except ValueError as e:
            errors.append(str(e))

    if update.llm_backend is not None or update.llm_model is not None or update.koboldcpp_url is not None:
        try:
            pipeline.set_llm_backend(
                backend=update.llm_backend or pipeline.llm_backend.value,
                model=update.llm_model,
                koboldcpp_url=update.koboldcpp_url,
            )
        except ValueError as e:
            errors.append(str(e))

    if update.system_prompt is not None:
        pipeline.set_system_prompt(update.system_prompt)

    config = pipeline.get_config()
    if errors:
        config["errors"] = errors

    return JSONResponse(content=config)


# ---------- Voices endpoint ----------

@app.get("/voices")
async def get_voices():
    return JSONResponse(content={
        "pocket_tts": POCKET_TTS_VOICES,
        "kokoro": ["af_heart"],
        "qwen3_voice_clone": ["profile-defined"],
    })
