# Jellyfin Whisper Worker

> **Automated, distributed subtitle generation daemon for Jellyfin media servers using OpenAI Whisper (`faster-whisper`) and local NVIDIA GPUs.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![CUDA 12 Ready](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## Overview

**Jellyfin Whisper Worker** is a lightweight, background daemon designed to transcribe and translate your entire Jellyfin library using a desktop or gaming laptop GPU (such as an NVIDIA RTX 2060 6GB) without overloading your primary server.

Instead of running slow CPU transcription on your Jellyfin server, this worker runs on any machine on your local network. It senses when you are idle (no keyboard/mouse input and GPU free), pulls audio streams from Jellyfin at >300x real-time speed, generates high-accuracy `.srt` subtitles using `faster-whisper` (`large-v3`), and uploads them straight back to Jellyfin via its REST API.

If you touch your mouse or start gaming, the worker **aborts within seconds** and releases all VRAM immediately.

```
       ┌────────────────────────┐
       │     Jellyfin Server    │
       │  (192.168.x.x:8096)    │
       └────┬──────────────▲────┘
            │              │
 Audio Stream (300x)    Upload Subtitle (.srt)
 /Audio/{id}/stream     POST /Videos/{id}/Subtitles
            │              │
       ┌────▼──────────────┴────┐
       │ Jellyfin Whisper Worker│
       │ (RTX 2060 / Idle Laptop│
       └────────────────────────┘
```

---

## Key Features

- **True Idle-Sensing**: Monitors Win32 user input (`GetLastInputInfo`) and GPU utilization (`nvidia-smi`). Only works when you aren't using the PC.
- **Instant Abort & VRAM Cleanup**: If you move your mouse or launch a game, extraction and transcription abort within seconds, deleting temp files and freeing all VRAM.
- **Fast Audio Extraction**: Streams audio directly via FFmpeg (`/Audio/{id}/stream`), extracting a 16 kHz mono WAV without downloading multi-gigabyte video files.
- **Whisper Subtitle Distinguishing**: Uploads subtitles tagged as `whisper - <Language>` (e.g., `whisper - English - SUBRIP - External`). In the Jellyfin player dropdown, Whisper tracks are clearly separated from embedded subtitles and third-party downloaded subtitles.
- **Intro Visual Badge**: Prepends an opening watermark (`[Subtitles: Whisper AI]`) during the first 3 seconds of the video, before dialogue starts.
- **Multi-Language Support**: Configurable comma-separated language queue (e.g. `eng,rum`). Transcribes native speech and translates to English automatically.
- **State Persistence**: Records completed items in an atomic `processed_state.json` tracker so titles are never needlessly re-transcribed across daemon restarts.
- **24/7 Daemon Operation**: Runs continuously in the background until manually stopped (`Ctrl+C`).
- **Auto CUDA DLL Discovery**: Automatically detects CUDA 12 runtime DLLs from Ollama (`%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12`) or official CUDA installations on Windows.

---

## Hardware & System Requirements

- **Operating System**: Windows 10/11 (fully supported with Win32 idle monitoring) or Linux.
- **GPU**: NVIDIA GPU with at least 6 GB VRAM (e.g. RTX 2060, RTX 3060, RTX 4060).
  - 6 GB VRAM: Use `WHISPER_MODEL=large-v3` with `COMPUTE_TYPE=int8` (consumes ~3.2–4.5 GB VRAM).
  - ≥8 GB VRAM: Use `COMPUTE_TYPE=float16`.
- **FFmpeg**: Automatically detected if installed on `PATH`, in Jellyfin Server directory (`C:\Program Files\Jellyfin\Server\ffmpeg.exe`), or custom path specified in `FFMPEG_PATH`.
- **Python**: Python 3.10 to 3.14.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ISmokeAir/Jellyfin-Whisper-Remote-Subtitles.git
cd Jellyfin-Whisper-Remote-Subtitles
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

*(Optional: If `torch` or `ctranslate2` cannot find `cublas64_12.dll`, ensure Ollama or CUDA Toolkit 12 is installed).*

### 3. Configure `.env`

Copy the template configuration file:

```bash
copy .env.example .env
```

Edit `.env` with your Jellyfin server details:

```ini
# Jellyfin Server URL
JELLYFIN_URL=http://192.168.1.100:8096

# Jellyfin API Key (generate under Dashboard -> API Keys)
JELLYFIN_API_KEY=your_jellyfin_api_key_here

# Idle detection threshold in seconds (default: 600s = 10 minutes)
IDLE_THRESHOLD_SECONDS=600

# Whisper model size (tiny, base, small, medium, large-v2, large-v3)
WHISPER_MODEL=large-v3

# Quantization (int8 for 6GB GPUs like RTX 2060, float16 for >=8GB GPUs)
COMPUTE_TYPE=int8

# Target subtitle languages (comma-separated ISO 639-2 codes)
# eng = translate to English
# rum/ron, spa, fra, deu, etc. = native language transcription
SUBTITLE_LANGS=eng,rum
```

---

## Usage

### 1. Check Library & Queue Status (`--status` / `--dry-run`)

Test your connection, inspect idle/GPU status, and view what has been processed vs queued next:

```bash
python main.py --dry-run
```

Output:
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
  Subtitle Languages : eng, rum

System Status:
  Idle Duration      : 340s (Threshold: 600s)
  GPU Utilization    : 4%
  Ready for Work     : No (Waiting for user idle & free GPU)

Checking Jellyfin library and queue progress...

Library Overview (845 total media titles):
  [OK] Completed by Whisper : 1 (0.1%)
  [...] Pending in Queue    : 844

Next Up in Queue:
   1. Grey's Anatomy S01E01 - A Hard Day's Night
      Needs Whisper tracks: eng, rum  [ID: 0a80da7374cb28df97478ae21e3bcda0]
   2. Scrubs S01E01 - My First Day
      Needs Whisper tracks: eng, rum  [ID: 388e73b250e3b29ffc06c1ea4d70f223]
```

---

### 2. Run the 24/7 Daemon

Double-click `start.bat` on Windows, or run from terminal:

```bash
python main.py
```

The daemon will:
1. Wait until user input has been idle for `IDLE_THRESHOLD_SECONDS` and GPU load is < 15%.
2. Fetch a batch of media items needing Whisper subtitles.
3. Lazily load the Whisper model into VRAM.
4. Transcribe each item segment by segment.
5. If you touch the keyboard or mouse, abort immediately and release all VRAM.
6. When work completes or when interrupted, sleep and repeat.

---

### 3. Additional CLI Flags

| Flag | Description |
| :--- | :--- |
| `--dry-run` / `--status` | Show system status, configuration, completed list, and pending queue without downloading media. |
| `--once` | Process one batch of items and exit cleanly (useful for scheduled tasks or testing). |
| `--ignore-idle` | Bypass keyboard/mouse and GPU idle checks (forces immediate processing). |
| `--item-id <ID>` | Process a single specific Jellyfin item by its GUID. |

Example (process a single movie immediately):
```bash
python main.py --ignore-idle --item-id 2d4aabc6e22df1870d8979fedbca9261
```

---

## How Subtitles Appear in Jellyfin

When selecting subtitles in the Jellyfin player (Web, Mobile, TV, Desktop):

| Track Type | Display Label in Jellyfin | File Stored on Server |
| :--- | :--- | :--- |
| **Whisper Subtitles** | **`whisper - English - SUBRIP - External`**<br>**`whisper - Romanian - SUBRIP - External`** | `<Title>.eng.whisper.srt`<br>`<Title>.rum.whisper.srt` |
| **External Web Subtitles** | `English - SUBRIP - External`<br>`Romanian - SUBRIP - External` | `<Title>.en.srt`<br>`<Title>.ro.srt` |
| **Embedded Container Subs** | `English - Default - SUBRIP`<br>`English SDH - SUBRIP` | Embedded in `.mkv` / `.mp4` |

---

## Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests -v
```

---

## Project Structure

```
├── .env.example          # Template configuration
├── .gitignore            # Excludes credentials, state, logs, and temp audio
├── config.py             # Configuration validation and environment loader
├── idle_monitor.py       # Win32 user input and nvidia-smi GPU monitor
├── jellyfin_client.py    # Jellyfin REST API and atomic state tracking
├── main.py               # Daemon loop, VRAM lifecycle, and CLI entry point
├── requirements.txt      # Python dependencies
├── start.bat             # One-click Windows startup script
├── transcriber.py        # FFmpeg audio extractor + faster-whisper engine
└── tests/
    └── test_worker.py    # Unit and integration test suite
```

---

## License

This project is licensed under the [MIT License](LICENSE).

