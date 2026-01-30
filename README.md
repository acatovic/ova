# Outrageous Voice Assistant

![Outrageous Logo](outrageous-logo-large.jpeg)

A **fully-local** voice assistant demo with a super simple FastAPI backend and a simple HTML front-end. All the models (ASR / LLM / TTS) are open weight and running locally, i.e. no data is being sent to the Internet nor any API. It's intended to demonstrate how easy it is to run a fully-local AI setup on affordable commodity hardware, while also demonstrating the uncanny valley and teasing out the ethical considerations of such a setup (see *Disclaimer and Ethical Considerations* at the bottom).


Models used:

* ASR: [NVIDIA parakeet-tdt-0.6b-v3 600M](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
* LLM: [Mistral ministral-3 3b 4-bit quantized](https://ollama.com/library/ministral-3:8b-instruct-2512-q4_K_M) (via Ollama), or any model via Koboldcpp
* TTS (Fast): [Hexgrad Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M)
* TTS (Pocket-TTS): [Kyutai Pocket-TTS 100M](https://github.com/kyutai-labs/pocket-tts) — 8 built-in voices, CPU-friendly

**Why "Outrageous"?** Because it was outrageously easy to create!

How it works:

1. Frontend captures user's audio (or typed text) and sends it to the backend
2. Backend parses the input, then:
   1. Transcribes the audio to text using an automatic speech recognition (ASR) model (skipped for typed text)
   2. Sends the transcribed text to the LLM, i.e. "the brain"
   3. Sends the LLM response to a text-to-speech (TTS) model
   4. Performs normalization of TTS output, converts it to bytes, and sends the bytes back to frontend
3. The frontend plays the response audio back to the user

On my system (RTX5070 12GiB VRAM), the whole round-trip-time using Kokoro is ~1 second.

When using "profiles", a custom system prompt is loaded to shape the assistant's personality.

## Demos

## Voice assistant with cloned voice TTS
https://github.com/user-attachments/assets/9b546ab1-8c71-44f2-85d8-433b3a3d267f

## Fast voice assistant with default TTS
https://github.com/user-attachments/assets/a296dbf7-9fa9-4904-bf22-d0cdc1e625a4

## Pre-requisites

- Python >=3.13
- [`uv`](https://docs.astral.sh/uv/) installed and available in PATH
- For Ollama LLM backend: [Ollama](https://ollama.com/) installed and running (`ollama` CLI available)
- For Koboldcpp LLM backend: [Koboldcpp](https://github.com/LostRuins/koboldcpp) running with a loaded model
- Verified on a system with RTX 5070 (12GiB VRAM); lower-end setups should be possible
- Pocket-TTS runs well on CPU only (no GPU required)

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/acatovic/ova_pocket.git
cd ova_pocket
```

### 2. Install

The install command downloads Python dependencies and only the models you need.

**Default setup** (Kokoro TTS + Ollama LLM):

```bash
./ova.sh install
```

**Pocket-TTS setup** (lightweight, 8 voices, CPU-friendly):

```bash
OVA_TTS_ENGINE=pocket_tts ./ova.sh install
```

**Koboldcpp setup** (no Ollama required):

```bash
OVA_LLM_BACKEND=koboldcpp ./ova.sh install
```

**Download everything** (all TTS engines + Ollama model):

```bash
./ova.sh install --all
```

### 3. Start

```bash
./ova.sh start
```

Open your browser to **http://localhost:8000** and start talking (or typing).

### 4. Stop

```bash
./ova.sh stop
```

## Docker (alternative)

You can run OVA in a Docker container instead of installing locally. Ollama or KoboldCpp still run on the host.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with [Compose V2](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU access)
- Ollama and/or KoboldCpp running on the host

### 1. Create a `.env` file

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN and any other settings
```

When Ollama runs on the host, the container reaches it through `host.docker.internal`:

```
OLLAMA_HOST=http://host.docker.internal:11434
```

For KoboldCpp on the host:

```
OVA_LLM_BACKEND=koboldcpp
OVA_KOBOLDCPP_URL=http://host.docker.internal:5001
```

### 2. Build and start

```bash
docker compose up --build
```

Open **http://localhost:8000** once both services are ready.

HuggingFace models are stored in a Docker volume (`hf-cache`) so they only download once.

### 3. Stop

```bash
docker compose down
```

## Configuration

All settings can be configured via environment variables at startup, and also changed live from the web UI settings bar.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OVA_PROFILE` | `default` | Voice profile (`default` or `dua`) |
| `OVA_TTS_ENGINE` | `kokoro` | TTS engine: `kokoro` or `pocket_tts` |
| `OVA_KOKORO_VOICE` | `af_heart` | Kokoro voice (see below) |
| `OVA_LLM_BACKEND` | `ollama` | LLM backend: `ollama` or `koboldcpp` |
| `OVA_LLM_MODEL` | auto | LLM model name (e.g. `mistral:latest`) |
| `OVA_KOBOLDCPP_URL` | `http://localhost:5001` | Koboldcpp API URL |
| `OVA_POCKET_TTS_VOICE` | `alba` | Pocket-TTS voice (see below) |

### Kokoro Voices

Kokoro ships with 28 English voices. Prefix key: `af` = American Female, `am` = American Male, `bf` = British Female, `bm` = British Male.

| Voice | Type |
|---|---|
| `af_heart` | American Female |
| `af_alloy` | American Female |
| `af_aoede` | American Female |
| `af_bella` | American Female |
| `af_jessica` | American Female |
| `af_kore` | American Female |
| `af_nicole` | American Female |
| `af_nova` | American Female |
| `af_river` | American Female |
| `af_sarah` | American Female |
| `af_sky` | American Female |
| `am_adam` | American Male |
| `am_echo` | American Male |
| `am_eric` | American Male |
| `am_fenrir` | American Male |
| `am_liam` | American Male |
| `am_michael` | American Male |
| `am_onyx` | American Male |
| `am_puck` | American Male |
| `am_santa` | American Male |
| `bf_alice` | British Female |
| `bf_emma` | British Female |
| `bf_isabella` | British Female |
| `bf_lily` | British Female |
| `bm_daniel` | British Male |
| `bm_fable` | British Male |
| `bm_george` | British Male |
| `bm_lewis` | British Male |

### Pocket-TTS Voices

Pocket-TTS ships with 8 built-in voices:

| Voice | Gender |
|---|---|
| `alba` | Female |
| `fantine` | Female |
| `cosette` | Female |
| `eponine` | Female |
| `azelma` | Female |
| `marius` | Male |
| `javert` | Male |
| `jean` | Male |

### Startup Examples

```bash
# Default fast TTS with Ollama
./ova.sh start

# Pocket-TTS with a specific voice
OVA_TTS_ENGINE=pocket_tts OVA_POCKET_TTS_VOICE=jean ./ova.sh start

# Voice cloning profile
OVA_PROFILE=dua ./ova.sh start

# Koboldcpp backend with a custom model
OVA_LLM_BACKEND=koboldcpp OVA_LLM_MODEL=my-model ./ova.sh start

# Koboldcpp on a remote machine
OVA_LLM_BACKEND=koboldcpp OVA_KOBOLDCPP_URL=http://192.168.1.100:5001 ./ova.sh start
```

### Web UI Settings

Once running, the web UI at http://localhost:8000 includes a settings bar where you can change all of the above live without restarting:

- **TTS Engine** dropdown (Kokoro / Pocket-TTS)
- **Voice** dropdown (populated based on selected TTS engine)
- **LLM Backend** dropdown (Ollama / Koboldcpp)
- **LLM Model** text field
- **Koboldcpp URL** text field (shown when Koboldcpp is selected)
- **Text input** box at the bottom for typing messages instead of speaking

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Send audio WAV bytes, receive audio WAV response |
| `/chat/text` | POST | Send JSON `{"text": "..."}`, receive audio WAV response |
| `/config` | GET | Get current configuration |
| `/config` | POST | Update configuration (TTS engine, voice, LLM backend, etc.) |
| `/voices` | GET | List available voices per TTS engine |
| `/health` | GET | GPU status, loaded models, and current config |

## Troubleshooting

Check service status:

```bash
./ova.sh status
```

Tail logs in real time:

```bash
./ova.sh logs
```

Check the backend health endpoint (while running):

```bash
curl -s http://localhost:5173/health | python3 -m json.tool
```

Clean reinstall if things break:

```bash
./ova.sh reinstall          # Linux / macOS
reinstall.bat               # Windows
```

Remove everything:

```bash
./ova.sh uninstall          # Linux / macOS
uninstall.bat               # Windows
```

**Enjoy!**

---

**Disclaimer & Ethical Considerations:** This project is a proof-of-concept demonstration and is provided **as is** without any warranties or guarantees. It is intended for educational and experimental purposes only. All this can be accomplished on a commodity PC that most people can afford.
