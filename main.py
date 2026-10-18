"""
main.py — Daemon loop, VRAM lifecycle management, and CLI entry point.

Usage:
    python main.py              # Run the daemon loop
    python main.py --dry-run    # Test config, show idle/GPU status, list items
    python main.py --once       # Process one batch then exit
"""

import argparse
import gc
import logging
import logging.handlers
import sys
import time
from pathlib import Path

import config
import idle_monitor
import jellyfin_client
import transcriber

# ── Logging Setup ────────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "whisper_worker.log"

# Root logger
_root = logging.getLogger()
_root.setLevel(logging.DEBUG)

# Console handler (INFO level)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
))
_root.addHandler(_console)

# File handler (DEBUG level, rotating 5 × 5 MB)
_file = logging.handlers.RotatingFileHandler(
    _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
_root.addHandler(_file)

logger = logging.getLogger("whisper-worker")

# Quiet down noisy libraries
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ── Constants ────────────────────────────────────────────────────────────────

POLL_INTERVAL_IDLE = 30      # seconds between idle checks
POLL_INTERVAL_ACTIVE = 60    # seconds to wait after user becomes active
BATCH_LIMIT = 5              # items to fetch per cycle


# ── CUDA Runtime DLL Setup ───────────────────────────────────────────────────

def _setup_cuda_dlls():
    """Ensure CUDA 12 runtime DLLs (cublas, cudart, etc.) can be loaded on Windows."""
    if sys.platform != "win32":
        return None
    import os
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12"),
        os.path.expandvars(r"%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"),
        os.path.expandvars(r"%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"),
        os.path.expandvars(r"%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"),
        os.path.expandvars(r"%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
        os.path.expandvars(r"%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "cublas64_12.dll")):
            try:
                os.add_dll_directory(p)
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            logger.debug("Added CUDA DLL directory: %s", p)
            return p
    return None

_setup_cuda_dlls()


# ── VRAM Lifecycle ───────────────────────────────────────────────────────────

def load_model():
    """Lazily load the Whisper model into VRAM and return it."""
    _setup_cuda_dlls()
    from faster_whisper import WhisperModel
    logger.info("Loading Whisper model '%s' onto CUDA (%s)...",
                config.WHISPER_MODEL, config.COMPUTE_TYPE)
    model = WhisperModel(
        config.WHISPER_MODEL,
        device="cuda",
        compute_type=config.COMPUTE_TYPE,
    )
    logger.info("Model loaded successfully")
    return model


def unload_model(model):
    """Explicitly free the model and reclaim VRAM."""
    if model is None:
        return
    logger.info("Unloading Whisper model and freeing VRAM...")
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass  # torch may not be directly installed
    logger.info("VRAM released")


# ── Dry Run & Status Overview ────────────────────────────────────────────────

def dry_run():
    """Test configuration, print status, and list completed vs pending queue."""
    print("=" * 65)
    print("  Jellyfin Whisper Worker - Status & Queue Overview")
    print("=" * 65)
    print()
    print("Configuration:")
    print(config.summary())
    print()

    # Idle / GPU status
    idle_secs = idle_monitor.get_idle_duration_seconds()
    gpu_util = idle_monitor.get_gpu_utilization()
    ready = idle_monitor.is_ready_for_work()
    print(f"System Status:")
    print(f"  Idle Duration      : {idle_secs:.0f}s (Threshold: {config.IDLE_THRESHOLD_SECONDS}s)")
    print(f"  GPU Utilization    : {gpu_util}%" if gpu_util is not None else "  GPU Utilization    : N/A")
    print(f"  Ready for Work     : {'Yes (Will transcribe when daemon runs)' if ready else 'No (Waiting for user idle & free GPU)'}")
    print()

    print("Checking Jellyfin library and queue progress...")
    try:
        summary = jellyfin_client.get_queue_summary(next_limit=10)
        total = summary["total_items"]
        done = summary["completed_count"]
        pending = summary["pending_count"]
        pct = (done / total * 100) if total else 0.0

        print(f"\nLibrary Overview ({total} total media titles):")
        print(f"  [OK] Completed by Whisper : {done} ({pct:.1f}%)")
        print(f"  [...] Pending in Queue    : {pending}")
        if summary["failed_count"]:
            print(f"  [X] Failed Items          : {summary['failed_count']}")
        if summary["skipped_count"]:
            print(f"  [-] Skipped (No Speech)   : {summary['skipped_count']}")

        completed_list = summary.get("completed_list", [])
        if completed_list:
            print(f"\nCompleted by Whisper:")
            for c in completed_list[-5:]:
                langs = ",".join(c.get("completed_langs", []))
                print(f"  * {c['name']} [{langs}] ({c['updated_at']})")

        queue = summary.get("next_in_queue", [])
        if queue:
            print(f"\nNext Up in Queue:")
            for i, q in enumerate(queue, 1):
                needs = ", ".join(q["needs_languages"])
                print(f"  {i:>2}. {q['name']}")
                print(f"      Needs Whisper tracks: {needs}  [ID: {q['id']}]")
        else:
            print("\nAll library titles are completed! The worker will monitor for new media.")
    except Exception as exc:
        print(f"  ERROR communicating with Jellyfin: {exc}")

    print()
    print("=" * 65)


# ── Main Daemon Loop ─────────────────────────────────────────────────────────

def run_daemon(once: bool = False, ignore_idle: bool = False, target_item_id: str | None = None):
    """
    Main loop:
      1. Wait until system is idle and GPU is free (unless ignore_idle=True).
      2. Query Jellyfin for items needing subtitles (or use target_item_id).
      3. Lazily load the Whisper model.
      4. Process each item; abort within seconds if user becomes active.
      5. Unload model and free all VRAM.
      Runs continuously 24/7 until manually stopped with Ctrl+C.
    """
    logger.info("Whisper worker daemon starting (running 24/7 until stopped)")
    logger.info("Config:\n%s", config.summary())

    model = None

    try:
        while True:
            # ── Phase 1: Wait for idle ───────────────────────────────────────
            if not ignore_idle:
                logger.info("Waiting for laptop to become idle (no mouse/keyboard and GPU free)...")
                while not idle_monitor.is_ready_for_work():
                    time.sleep(POLL_INTERVAL_IDLE)
            else:
                logger.info("Bypassing idle check (--ignore-idle active)")

            # ── Phase 2: Query Jellyfin ──────────────────────────────────────
            if target_item_id:
                logger.info("Fetching target item: %s", target_item_id)
                try:
                    items = [jellyfin_client.get_item(target_item_id)]
                except Exception as exc:
                    logger.error("Failed to fetch item %s: %s", target_item_id, exc)
                    break
            else:
                try:
                    items = jellyfin_client.get_unsubtitled_items(limit=BATCH_LIMIT)
                except Exception as exc:
                    logger.error("Failed to query Jellyfin (will retry): %s", exc)
                    time.sleep(POLL_INTERVAL_ACTIVE)
                    continue

            if not items:
                logger.info("All scanned items are up to date! Sleeping %ds before checking for new media...",
                            POLL_INTERVAL_ACTIVE)
                time.sleep(POLL_INTERVAL_ACTIVE)
                if once or target_item_id:
                    logger.info("Exiting after empty queue")
                    break
                continue

            # ── Phase 3: Load model (lazy) ───────────────────────────────────
            try:
                if model is None:
                    model = load_model()
            except Exception as exc:
                logger.error("Failed to load Whisper model: %s", exc)
                time.sleep(POLL_INTERVAL_ACTIVE)
                continue

            # ── Phase 4: Process items ───────────────────────────────────────
            _LANG_MAP = {
                "eng": "en", "rum": "ro", "ron": "ro", "spa": "es", "fra": "fr",
                "fre": "fr", "deu": "de", "ger": "de", "ita": "it", "por": "pt",
                "jpn": "ja", "kor": "ko", "zho": "zh", "chi": "zh", "ara": "ar",
                "hin": "hi", "rus": "ru", "pol": "pl", "nld": "nl", "dut": "nl",
                "tur": "tr", "hun": "hu", "ces": "cs", "cze": "cs", "swe": "sv",
                "dan": "da", "fin": "fi", "nor": "no", "ukr": "uk", "bul": "bg",
                "hrv": "hr", "srp": "sr", "slk": "sk", "slo": "sk",
            }

            aborted = False
            for item in items:
                item_id = item["Id"]
                display = jellyfin_client.get_item_display_name(item)
                item_type = item.get("Type", "Media")
                logger.info("Processing: %s [%s]", display, item_id)

                # Double-check idle state before starting each item
                if not ignore_idle and not idle_monitor.is_ready_for_work():
                    logger.warning("System no longer idle — aborting batch")
                    aborted = True
                    break

                stream_url = jellyfin_client.get_audio_stream_url(item_id)
                any_success = False
                successful_langs: list[str] = []
                target_langs = item.get("_missing_languages", config.SUBTITLE_LANGS)

                for lang_code in target_langs:
                    short = _LANG_MAP.get(lang_code, lang_code)

                    # "eng" = translate to English; anything else = transcribe natively
                    if lang_code == "eng":
                        task = "translate"
                        whisper_lang = None  # auto-detect source language
                    else:
                        task = "transcribe"
                        whisper_lang = short  # force transcription in this language

                    logger.info("  → Generating %s subtitle (task=%s, lang=%s)",
                                lang_code, task, whisper_lang or "auto")

                    result = transcriber.process_item(
                        stream_url, model, language=whisper_lang, task=task,
                        check_idle=not ignore_idle,
                    )

                    if result.aborted:
                        logger.warning("Transcription aborted for %s — user active", display)
                        aborted = True
                        break

                    if not result.success:
                        logger.error("Transcription failed for %s (%s): %s",
                                     display, lang_code, result.error)
                        continue

                    if not result.srt_text:
                        logger.info("No speech for %s (%s) — skipping this track",
                                    display, lang_code)
                        continue

                    # Upload this language's subtitle track
                    try:
                        jellyfin_client.upload_subtitle(
                            item_id, result.srt_text, language=lang_code,
                        )
                        logger.info("Uploaded %s subtitle for %s (%d segments)",
                                    lang_code, display, result.segment_count)
                        any_success = True
                        successful_langs.append(lang_code)

                        # Update state immediately so partial work is preserved
                        jellyfin_client.mark_processed(
                            item_id=item_id,
                            status="in_progress",
                            name=display,
                            item_type=item_type,
                            completed_langs=[lang_code],
                            detail=f"{lang_code}: {result.segment_count} segments",
                        )
                    except Exception as exc:
                        logger.error("Upload failed for %s (%s): %s",
                                     display, lang_code, exc)

                if aborted:
                    break

                # Mark overall item status
                pending_still = jellyfin_client.get_item_pending_languages(item_id)
                if not pending_still:
                    jellyfin_client.mark_processed(
                        item_id=item_id,
                        status="success",
                        name=display,
                        item_type=item_type,
                        detail=f"All {len(config.SUBTITLE_LANGS)} tracks complete",
                    )
                elif not any_success:
                    jellyfin_client.mark_processed(
                        item_id=item_id,
                        status="skipped",
                        name=display,
                        item_type=item_type,
                        detail="No speech detected or all tracks failed",
                    )

            # ── Phase 5: Unload model ────────────────────────────────────────
            unload_model(model)
            model = None

            if aborted:
                logger.info("Batch aborted — waiting for system to become idle again")

            if once or target_item_id:
                logger.info("Task complete — exiting as requested")
                break

            # Brief cooldown before next cycle
            time.sleep(POLL_INTERVAL_IDLE)

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user (Ctrl+C). Cleaning up...")
        if model is not None:
            unload_model(model)
        logger.info("Daemon cleanly stopped.")
        sys.exit(0)


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Jellyfin Whisper Worker — automatic local subtitle generation daemon"
    )
    parser.add_argument(
        "--dry-run",
        "--status",
        action="store_true",
        dest="dry_run",
        help="Display library overview, what has been done, and next queue items",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one batch of items then exit (don't loop 24/7)",
    )
    parser.add_argument(
        "--ignore-idle",
        action="store_true",
        help="Bypass idle and user-activity checks (for manual testing)",
    )
    parser.add_argument(
        "--item-id",
        type=str,
        default=None,
        help="Process a single Jellyfin item ID directly",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        sys.exit(0)

    run_daemon(once=args.once, ignore_idle=args.ignore_idle, target_item_id=args.item_id)


if __name__ == "__main__":
    main()

