# Outrageous Voice Assistant

![Outrageous Logo](outrageous-logo-large.jpeg)

A local voice assistant demo with a FastAPI backend and a simple HTML front-end. All the models (ASR / LLM / TTS) are open weight and running locally.

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
