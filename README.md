# Outrageous Voice Assistant

![Outrageous Logo](outrageous-logo-large.jpeg)

A local voice assistant demo with a FastAPI backend and a simple HTML front-end. All the models (ASR / LLM / TTS) are open weight and running locally.

Models used:

* ASR: [NVIDIA parakeet-tdt-0.6b-v3 600M](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
* LLM: [Mistral ministral-3 8b 4-bit quantized](https://ollama.com/library/ministral-3:8b-instruct-2512-q4_K_M)
* TTS: [Hexgrad Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M)

How it works:

```mermaid
sequenceDiagram
  autonumber
  participant FE as Frontend (UI / Client)
  participant BE as Backend (API)
  participant ASR as Local ASR Model
  participant LLM as Local LLM
  participant TTS as Local TTS Model

  Note over FE,BE: Audio input -> response audio (all models run locally)
  FE->>BE: HTTP POST /chat (wave bytes)
  activate BE

  BE->>BE: Receive wave bytes
  BE->>BE: Parse header (channels, sample rate, bit depth)
  alt Resampling needed?
    BE->>BE: Resample audio (optional)
  end
  BE->>BE: Convert audio samples -> tensor

  BE->>ASR: Transcribe(tensor)
  activate ASR
  ASR-->>BE: transcript (text)
  deactivate ASR

  BE->>LLM: Generate(system prompt + transcript)
  activate LLM
  LLM-->>BE: response text
  deactivate LLM

  BE->>TTS: Synthesize(response text)
  activate TTS
  TTS-->>BE: audio output (samples)
  deactivate TTS

  BE->>BE: Encode samples -> wave bytes
  BE-->>FE: HTTP 200 (wave bytes)
  deactivate BE

  FE->>FE: Play audio to user
```

## Demo

[Watch the demo video](ova-demo.mp4)

## Pre-requisites

- Python >=3.13
- `uv` installed and available in PATH
- Ollama installed and running (`ollama` CLI available)

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
