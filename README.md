# Outrageous Voice Assistant

![Outrageous Logo](outrageous-logo-large.jpeg)

A **fully-local** voice assistant demo with a super simple FastAPI backend and a simple HTML front-end. All the models (ASR / LLM / TTS) are open weight and running locally, i.e. no data is being sent to the Internet nor any API. It's intended to demonstrate how easy it is to run a fully-local AI setup on affordable commodity hardware, while also demonstrating the uncanny valley and teasing out the ethical considerations of such a setup (see *Disclaimer and Ethical Considerations* at the bottom).


Models used:

* ASR: [NVIDIA parakeet-tdt-0.6b-v3 600M](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
* LLM: [Mistral ministral-3 3b 4-bit quantized](https://ollama.com/library/ministral-3:8b-instruct-2512-q4_K_M)
* TTS (Simple): [Hexgrad Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M)
* TTS (With Voice Cloning): [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

**Why "Outrageous"?** Because it was outrageously easy to create!

How it works:

1. Frontend captures user's audio and sends a blob of bytes to the backend `/chat` endpoint
2. Backend parses the bytes, extracts sample rate (SR) and channels, then:
   1. Transcribes the audio to text using an automatic speech recognition (ASR) model
   2. Sends the transcribed text to the LLM, i.e. "the brain"
   3. Sends the LLM response to a text-to-speech (TTS) model
   4. Performs normalization of TTS output, converts it to bytes, and sends the bytes back to frontend
3. The frontend plays the response audio back to the user

On my system (RTX5070 12GiB VRAM), the whole round-trip-time using Kokoro is ~1 second.

When using "profiles" (or voice cloning), there is an additional pre-step where a 3-5 second `wav` clip with a corresponding transcription and a prompt, is used for TTS. This leverages Qwen3-TTS and doesn't require any finetuning. Note however that responses will be slightly slower. The reason I included Kokoro as a default / non-voice cloning TTS is (1) it's very fast, and (2) I really like the quality of the default voice.


## TO-DO

- [ ] Add Voice Activity Detection (VAD) client-side (e.g. see  https://docs.vad.ricky0123.com/)
- [ ] Remove the need for prepared transcript for voice cloning - use ASR as part of a "warmup"
- [ ] Add orchestration to detect more complex tasks (e.g. requiring GPT/Claude) or whether tool calls (web search, file search, other cli/API calls are needed)


## Demos

## Voice assistant with Dua Lipa's cloned voice
https://github.com/user-attachments/assets/9b546ab1-8c71-44f2-85d8-433b3a3d267f

## Voice assistant with the default voice
https://github.com/user-attachments/assets/a296dbf7-9fa9-4904-bf22-d0cdc1e625a4

## Pre-requisites

- Python >=3.12
- `uv` installed and available in PATH
- Ollama installed and running (`ollama` CLI available)
- Verified on a system with RTX 5070 (12GiB VRAM); lower-end setups should be possible

## Install

Fetch Python deps and HF/Ollama models:

```bash
./ova.sh install
```

## Start

Start the front-end and back-end services (non-blocking) with a fast default voice assistant:

```bash
./ova.sh start
```

To start the voice assistant with one of the pre-cloned voices (NOTE: response time will be slower):

```bash
OVA_PROFILE=dua ./ova.sh start  # NOTE: with cloned voice of a famous artist
```

- Front-end: http://localhost:8000
- Back-end: http://localhost:5173

Logs and PIDs are stored under `.ova/`. If you want to follow the logs in another terminal window:

```bash
tail -f .ova/backend.log
```

## Stop

Stop all services:

```bash
./ova.sh stop
```

## Adding new voices (clones / profiles)

In order to add a new voice, no code changes are required. You simply need to do the following:

1. Create a new directory `profiles/<voice>/`
2. Add a 3-5 second voice clip `ref_audio.wav`, a direct transcription of that clip `ref_text.txt`, and any instructions in the `prompt.txt` - all under the sub-directory created in the previous step.
3. To start the service with the new voice, simply run `OVA_PROFILE=<voice> ./ova.sh start`

And that's it!

**Enjoy!**

---

**Disclaimer & Ethical Considerations:** This project is a proof-of-concept demonstration and is provided **as is** without any warranties or guarantees. It is intended for educational and experimental purposes only. The voice cloning is also purely for educational purposes - for real-life/commercial use, one should always seek the relevant permissions. This demo also highlights the ethical and security aspects - the ease with which one can clone a voice with no finetuning, using only a 3-5 second audio clip - which is both eerie, and potentially dangerous in the wrong hands. All this can be accomplished on a commodity PC that most people can afford.
