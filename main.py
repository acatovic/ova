import io
import wave

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
import nemo.collections.asr as nemo_asr
import numpy as np
from ollama import chat
import soxr
import torch


SR = 24000

# 'a' => US/American English
tts_pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M")

chat_model = "ministral-3:3b-instruct-2512-q4_K_M"

# Initialize ASR model
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

CHAT_SYSTEM_PROMPT = """
You are a helpful assistant.
You respond with short, elegant, and concise answers.

When responding **ALWAYS** follow these instructions:
  - Be concise and to the point - prioritize conciseness and polish over verbosity.
  - **NEVER** respond in bullet points - use proper sentences.
  - **DO NOT** include any Markdown formatting, asterisks, underscores, hashes, or other formatting.
  - **DO NOT** include emojis.
"""

context = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]


def resample(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    if src_sr == dst_sr or arr.size == 0:
        return arr
    
    return soxr.resample(arr, src_sr, dst_sr, quality="HQ")


def rms_normalize(arr: np.ndarray, target_rms=0.15, peak_limit=0.90, eps=1e-12) -> np.ndarray:
    x = arr.astype(np.float32)

    rms = np.sqrt(np.mean(x * x) + eps)
    if rms < eps:
        return x  # silence

    x = x * (target_rms / rms)

    # prevent clipping
    peak = np.max(np.abs(x)) + eps
    if peak > peak_limit:
        x = x * (peak_limit / peak)

    return x


def transcribe(asr_model, wav_bytes: bytes) -> str:
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
    model_sr = int(getattr(getattr(asr_model, "cfg", None), "sample_rate", SR))
    audio = resample(audio, src_sr, model_sr)

    # Torch tensors on model device
    device = next(asr_model.parameters()).device
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, T]
    length_tensor = torch.tensor([audio.shape[0]], device=device, dtype=torch.long)

    asr_model.eval()
    with torch.inference_mode():
        out = asr_model(input_signal=audio_tensor, input_signal_length=length_tensor)

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, logit_lengths = out[0], out[1]
        elif isinstance(out, dict):
            logits = out.get("logits", out.get("encoded"))
            logit_lengths = out.get("logit_lengths", out.get("encoded_len"))
            if logits is None or logit_lengths is None:
                raise RuntimeError(f"Unexpected model output keys: {list(out.keys())}")
        else:
            raise RuntimeError(f"Unexpected model output type: {type(out)}")

        decoding = getattr(asr_model, "decoding", None)
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



def numpy_to_wav_bytes(arr: np.ndarray, sr: int) -> bytes:
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
        arr = np.clip(arr, -1.0, 1.0)
    
    # RMS normalize
    arr = rms_normalize(arr)

    arr = np.clip(arr, -1.0, 1.0)
    arr_i16 = (arr * 32767.0).astype(np.int16)

    if arr_i16.ndim == 1:
        arr_i16 = arr_i16[:, None]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(arr_i16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr_i16.tobytes())

    return buf.getvalue()


def _chat(text: str) -> str:
    context.append({"role": "user", "content": text})
    response = chat(
        model=chat_model,
        messages=context,
        think=False,
        stream=False,
    )

    response = response.message.content.replace("**", "").replace("_", "").replace("__", "").replace("#", "").strip()
    context.append({"role": "assistant", "content": response})

    return response


#####################
# APP CONFIGURATION #
#####################

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_class=Response)
async def chat_request_handler(request: Request):
    audio_bytes = await request.body()
    transcribed_text = transcribe(asr_model, audio_bytes)
    
    if not transcribed_text:
        # Return empty audio if no transcription
        empty_arr = np.array([], dtype=np.float32)
        wav_bytes = numpy_to_wav_bytes(empty_arr, SR)
        return Response(content=wav_bytes, media_type="audio/wav")
        
    # Get LLM response
    llm_response = _chat(transcribed_text)
        
    # Generate TTS audio
    generator = tts_pipeline(llm_response, voice='af_heart')
        
    chunks = []
    for _, _, audio in generator:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size > 0:
            chunks.append(audio)
        
    arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    wav_bytes = numpy_to_wav_bytes(arr, SR)
    return Response(content=wav_bytes, media_type="audio/wav")
