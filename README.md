# Outrageous Voice Assistant

![Outrageous Logo](outrageous-logo-large.jpeg)

A local voice assistant demo with a FastAPI backend and a simple HTML front-end. All the models (ASR / LLM / TTS) are open weight and running locally.

Models used:

* ASR: [NVIDIA parakeet-tdt-0.6b-v3 600M](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
* LLM: [Mistral ministral-3 3b 4-bit quantized](https://ollama.com/library/ministral-3:8b-instruct-2512-q4_K_M)
* TTS: [Hexgrad Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M)

**Why "Outrageous"?** Because it was outrageously easy to create!

How it works:

1. Frontend captures user's audio and sends a blob of bytes to the backend `/chat` endpoint
2. Backend parses the bytes, extracts sample rate (SR) and channels, then:
   1. Transcribes the audio to text using an automatic speech recognition (ASR) model
   2. Sends the transcribed text to the LLM, i.e. "the brain"
   3. Sends the LLM response to a text-to-speech (TTS) model
   4. Performs normalization of TTS output, converts it to bytes, and sends the bytes back to frontend
3. The frontend plays the response audio back to the user

On my system (RTX5070 12GiB VRAM), the whole round-trip-time is <2 seconds.

## Demo

https://github.com/user-attachments/assets/a296dbf7-9fa9-4904-bf22-d0cdc1e625a4

## Pre-requisites

- Python >=3.13
- `uv` installed and available in PATH
- Ollama installed and running (`ollama` CLI available)
- Verified on a system with RTX 5070 (12GiB VRAM); lower-end setups should be possible

## Install

Fetch Python deps and HF/Ollama models:

```
./ova install
```

## Start

Start the front-end and back-end services (non-blocking):

```
./ova start
```

- Front-end: http://localhost:8000
- Back-end: http://localhost:5173

Logs and PIDs are stored under `.ova/`.

## Stop

Stop all services:

```
./ova stop
```

**Enjoy!**

---

**Disclaimer:** This project is a proof-of-concept demonstration and is provided "as is" without any warranties or guarantees. It is intended for educational and experimental purposes only. Use at your own risk.
