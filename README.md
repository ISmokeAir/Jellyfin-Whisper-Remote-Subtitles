# Jellyfin Whisper Worker

Background subtitle generation daemon for Jellyfin media servers using faster-whisper and Meta NLLB-200 on local NVIDIA GPUs.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![CUDA 12 Ready](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Overview

Jellyfin Whisper Worker offloads speech-to-text and subtitle translation from your primary Jellyfin server to a local network workstation or gaming PC equipped with an NVIDIA GPU.

The daemon monitors system idle state and GPU utilization. When the host machine is inactive, it extracts audio streams over HTTP from Jellyfin, performs speech recognition using `faster-whisper`, translates text using a local Meta NLLB-200 model, and uploads the resulting `.srt` subtitle tracks back to the server via the Jellyfin REST API.

If user interaction (mouse movement or keystrokes) is detected during processing, the daemon aborts execution within seconds, purges temporary files, and frees all GPU VRAM.

```
       ┌────────────────────────┐
       │     Jellyfin Server    │
       │  (http://host:8096)    │
       └────┬──────────────▲────┘
            │              │
  HTTP Audio Stream        Upload Subtitle (.srt)
  /Audio/{id}/stream       POST /Videos/{id}/Subtitles
            │              │
       ┌────▼──────────────┴────┐
       │ Jellyfin Whisper Worker│
       │ (NVIDIA GPU / CUDA 12) │
       │                        │
       │ 1. faster-whisper      │
       │    (large-v3, INT8)    │
       │ 2. Meta NLLB-200       │
       │    (Seq2Seq MT, INT8)  │
       └────────────────────────┘
```

---

## Architecture & Pipeline

1. **Idle and Activity Detection**
   - Windows: Win32 API (`GetLastInputInfo`) monitors elapsed time since last physical keyboard/mouse input.
   - GPU utilization is polled via `pynvml` or `nvidia-smi`.
   - The daemon initiates jobs only when idle duration exceeds `IDLE_THRESHOLD_SECONDS` and GPU load is under 15%.
   - Active user input immediately terminates active child processes and model inference loops.

2. **Direct Stream Audio Extraction**
   - FFmpeg connects directly to Jellyfin's `/Audio/{id}/stream` endpoint using API token authentication.
   - Extracts a single-channel 16 kHz 16-bit PCM WAV without downloading full video containers.

3. **Speech Recognition & Auto-Detection**
   - Inference runs via `faster-whisper` (`CTranslate2` backend).
   - Audio language is detected automatically from the first 30 seconds of audio (`task="transcribe", language=None`).
   - Silences are filtered via Silero VAD (`vad_filter=True`).
   - Anti-hallucination controls: `condition_on_previous_text=False`, `repetition_penalty=1.1`.
   - If the audio is non-English, Whisper optionally generates an English translation track (`task="translate"`).

4. **Local Neural Translation (Meta NLLB-200)**
   - Target languages not covered by native transcription (such as Romanian `ron`) are translated using Meta's `nllb-200-distilled-600M` model running in INT8 via CTranslate2.
   - Subtitle blocks are parsed, batched, and translated with preserved timestamps.
   - The opening watermark badge is localized automatically (e.g., `[Subtitrări: Whisper AI]`).
   - Operates 100% offline with zero external API dependencies.

5. **Subtitle Ingestion & Track Labeling**
   - Subtitles are base64-encoded and uploaded to Jellyfin at `/Videos/{id}/Subtitles`.
   - The language field is tagged as `<lang>.whisper` (e.g., `eng.whisper`, `ron.whisper`).
   - Jellyfin renders these tracks as `whisper - English` and `whisper - Romanian` in client players, allowing users to distinguish AI-generated subtitles from external web downloads or embedded container tracks.

6. **State Tracking & VRAM Reclamation**
   - State is stored atomically in `processed_state.json`.
   - Completed titles and languages are recorded so files are not re-processed across restarts.
   - Whisper and NLLB models are explicitly deleted, triggering garbage collection and CUDA cache clearance when transitioning back to idle or active states.

---

## Hardware Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with 6 GB VRAM (e.g., GTX 1660 Ti, RTX 2060) | NVIDIA GPU with $\ge$8 GB VRAM (e.g., RTX 3060, RTX 4060) |
| **Compute Configuration** | `WHISPER_MODEL=large-v3`, `COMPUTE_TYPE=int8` (~3.5 GB VRAM) | `WHISPER_MODEL=large-v3`, `COMPUTE_TYPE=float16` (~5.5 GB VRAM) |
| **RAM** | 8 GB System RAM | 16 GB System RAM |
| **OS** | Windows 10/11 (x64) or Linux (x64) | Windows 11 (x64) |
| **Python** | Python 3.10 - 3.14 (64-bit) | Python 3.11 - 3.14 (64-bit) |
| **FFmpeg** | FFmpeg 5.x, 6.x, or 7.x | Installed on `PATH` or bundled with Jellyfin Server |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ISmokeAir/Jellyfin-Whisper-Remote-Subtitles.git
cd Jellyfin-Whisper-Remote-Subtitles
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. CUDA 12 Runtime Setup (Windows)

The worker automatically detects CUDA 12 runtime DLLs (`cublas64_12.dll`) from standard locations, including:
- Local Ollama installations (`%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12`)
- NVIDIA CUDA Toolkit (`%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`)

If CUDA DLLs are located in a custom path, add the directory to your system `PATH`.

### 4. Configuration

Copy the example environment configuration:

```bash
copy .env.example .env
```

Configure the following variables in `.env`:

```ini
# Jellyfin Server URL
JELLYFIN_URL=http://192.168.1.100:8096

# Jellyfin API Key (Dashboard -> API Keys)
JELLYFIN_API_KEY=your_api_key_here

# Idle threshold before worker starts (in seconds)
IDLE_THRESHOLD_SECONDS=600

# Whisper model size (tiny, base, small, medium, large-v2, large-v3)
WHISPER_MODEL=large-v3

# Quantization type (int8 for 6GB GPUs, float16 for >=8GB GPUs)
COMPUTE_TYPE=int8

# Target subtitle languages (comma-separated ISO 639-2 codes)
SUBTITLE_LANGS=eng,ron

# Default subtitle track to flag as active/default in player (e.g., eng or ron)
DEFAULT_SUBTITLE_LANG=eng

# Enable dynamic audio language auto-detection
AUTO_DETECT_LANGUAGE=true

# Local Hugging Face repository for NLLB translation
TRANSLATION_MODEL=mijuanlo/nllb-200-distilled-600M-ct2-int8
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `JELLYFIN_URL` | String | *(Required)* | Base URL of the Jellyfin server including protocol and port. |
| `JELLYFIN_API_KEY` | String | *(Required)* | Authentication token generated in the Jellyfin Dashboard. |
| `IDLE_THRESHOLD_SECONDS` | Integer | `600` | Inactivity duration required before background processing begins. |
| `WHISPER_MODEL` | String | `large-v3` | Model identifier (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`). |
| `COMPUTE_TYPE` | String | `int8` | Quantization precision (`int8`, `int8_float16`, `float16`, `float32`). |
| `SUBTITLE_LANGS` | String | `eng,ron` | Desired subtitle tracks per media item. Supports ISO 639-1 and 639-2 codes. |
| `DEFAULT_SUBTITLE_LANG` | String | `eng` | Language to tag as `Default` in Jellyfin, making it auto-select during playback. |
| `AUTO_DETECT_LANGUAGE` | Boolean | `true` | Automatically detect spoken language per title instead of forcing a fixed language. |
| `TRANSLATION_MODEL` | String | `mijuanlo/...` | Hugging Face model repository for CTranslate2 INT8 NLLB-200. |

---

## Usage

### 1. Library & Status Overview

Inspect server connection, idle status, and queue progress:

```bash
python main.py --dry-run
```

Example output:

```text
=================================================================
  Jellyfin Whisper Worker - Status & Queue Overview
=================================================================

Configuration:
  Jellyfin URL       : http://192.168.1.100:8096
  API Key            : 2283****dcd8
  Idle Threshold     : 600s
  Whisper Model      : large-v3
  Compute Type       : int8
  Subtitle Languages : eng, ron
  Auto-Detect Audio  : True
  Translation Model  : mijuanlo/nllb-200-distilled-600M-ct2-int8

System Status:
  Idle Duration      : 420s (Threshold: 600s)
  GPU Utilization    : 2%
  Ready for Work     : No (Waiting for user idle & free GPU)

Checking Jellyfin library and queue progress...

Library Overview (845 total media titles):
  [OK] Completed by Whisper : 2 (0.2%)
  [...] Pending in Queue    : 843

Completed by Whisper:
  * Over the Garden Wall S01E01 - The Old Grist Mill [eng,ron] (2026-10-18T22:33:45)
  * Grey's Anatomy S01E01 - A Hard Day's Night [eng,ron] (2026-10-18T22:49:58)

Next Up in Queue:
   1. My Babysitter's a Vampire S01E01 - Lawn of the Dead
      Needs Whisper tracks: eng, ron  [ID: 7311d35aba6ee6110d11674fe34e4a90]
   2. Scrubs S01E01 - My First Day
      Needs Whisper tracks: eng, ron  [ID: 388e73b250e3b29ffc06c1ea4d70f223]
```

### 2. Run the Background Daemon

Launch the daemon in the foreground or via startup script:

```bash
# Windows batch launcher
start.bat

# Standard CLI invocation
python main.py
```

The daemon will:
1. Poll system idle duration and GPU load every 30 seconds.
2. Query Jellyfin for pending titles when idle conditions are met.
3. Stream audio, transcribe native speech, and translate target tracks.
4. Upload `.srt` tracks to Jellyfin and update `processed_state.json`.
5. Release VRAM immediately upon user activity or task completion.

### 3. CLI Arguments

| Argument | Description |
| :--- | :--- |
| `--dry-run`, `--status` | Display system status, configuration, and queue statistics without processing media. |
| `--once` | Execute a single batch of items and terminate cleanly upon completion. |
| `--ignore-idle` | Bypass keyboard, mouse, and GPU idle checks (recommended for testing or dedicated headless servers). |
| `--item-id <ID>` | Process a single Jellyfin media item directly by its GUID. |

Example: Process a specific episode immediately:

```bash
python main.py --ignore-idle --item-id 0a80da7374cb28df97478ae21e3bcda0
```

---

## Client Subtitle Display

Subtitles uploaded by the daemon are distinguished in the Jellyfin audio/subtitle selector:

| Origin | Player Label | Track File |
| :--- | :--- | :--- |
| **Whisper Worker (Original)** | `whisper - English - SUBRIP - External` | `<Title>.eng.whisper.srt` |
| **Whisper Worker (Translated)** | `whisper - Romanian - SUBRIP - External` | `<Title>.ron.whisper.srt` |
| **Third-Party External Subtitles** | `English - SUBRIP - External` | `<Title>.en.srt` |
| **Embedded Container Streams** | `English - Default - SUBRIP` | Embedded in MKV/MP4 |

---

## Verification & Testing

Execute the test suite to verify configuration parsing, language normalization, timestamp formatting, and subtitle translation logic:

```bash
python -m unittest discover -s tests -v
```

---

## Project Structure

```
├── .env.example          # Template environment configuration
├── .gitignore            # Excludes credentials, logs, state files, and cache
├── config.py             # Environment configuration validation and loader
├── idle_monitor.py       # Win32 input monitoring and GPU utilization poller
├── jellyfin_client.py    # Jellyfin REST API integration and state persistence
├── LICENSE               # MIT License
├── main.py               # Daemon loop, lifecycle management, and CLI entry point
├── README.md             # Project documentation
├── requirements.txt      # Production dependencies
├── start.bat             # Windows batch startup script
├── transcriber.py        # FFmpeg stream extractor and faster-whisper pipeline
├── translator.py         # Meta NLLB-200 local translation engine
└── tests/
    └── test_worker.py    # Unit and regression test suite
```

---

## License

This project is licensed under the [MIT License](LICENSE).
