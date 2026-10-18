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
import os
import sys
import time
from pathlib import Path

import config
import idle_monitor
import jellyfin_client
import transcriber
import translator

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
    """Explicitly free Whisper and translator models and reclaim all VRAM."""
    if model is not None:
        logger.info("Unloading Whisper model and freeing VRAM...")
        del model
    translator.unload_translator()
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
                target_langs = [
                    jellyfin_client.normalize_to_3letter(l)
                    for l in item.get("_missing_languages", config.SUBTITLE_LANGS)
                ]
                logger.info("  Target subtitle tracks needed: %s", target_langs)

                import tempfile
                tmp_dir = Path(__file__).parent
                fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
                os.close(fd)

                successful_langs: list[str] = []
                generated_srts: dict[str, str] = {}

                try:
                    # Stage 1: Single Audio Extraction
                    logger.info("  [1/3] Extracting audio for %s...", display)
                    if not transcriber.extract_audio(stream_url, wav_path, check_idle=not ignore_idle):
                        if not ignore_idle and idle_monitor.get_idle_duration_seconds() < 10:
                            logger.warning("Audio extraction aborted due to user activity")
                            aborted = True
                            break
                        logger.error("Failed to extract audio for %s", display)
                        continue

                    # Stage 2: Whisper Native Transcription with Auto-Detection
                    logger.info("  [2/3] Transcribing audio with Whisper (auto-detecting language)...")
                    native_res = transcriber.transcribe(
                        wav_path, model, language=None, task="transcribe", check_idle=not ignore_idle
                    )

                    if native_res.aborted:
                        logger.warning("Transcription aborted for %s — user active", display)
                        aborted = True
                        break

                    if not native_res.success:
                        logger.error("Transcription failed for %s: %s", display, native_res.error)
                        continue

                    if not native_res.srt_text:
                        logger.info("No speech detected for %s — skipping item", display)
                        jellyfin_client.mark_processed(
                            item_id=item_id, status="skipped", name=display, item_type=item_type,
                            detail="No speech detected (instrumental or silent)",
                        )
                        continue

                    detected_short = native_res.detected_language or "en"
                    detected_3 = jellyfin_client.normalize_to_3letter(detected_short)
                    logger.info("  Detected audio language: '%s' (%s) [confidence %.2f]",
                                detected_short, detected_3, native_res.language_probability)

                    # Store native transcript
                    generated_srts[detected_3] = native_res.srt_text

                    # If the native language is one of the desired tracks, upload it directly
                    if detected_3 in target_langs:
                        try:
                            jellyfin_client.upload_subtitle(item_id, native_res.srt_text, language=detected_3)
                            logger.info("  Uploaded native %s subtitle for %s (%d segments)",
                                        detected_3, display, native_res.segment_count)
                            successful_langs.append(detected_3)
                            jellyfin_client.mark_processed(
                                item_id=item_id,
                                status="in_progress",
                                name=display,
                                item_type=item_type,
                                completed_langs=[detected_3],
                                detail=f"Native {detected_3}: {native_res.segment_count} segments",
                            )
                        except Exception as exc:
                            logger.error("Upload failed for %s (%s): %s", display, detected_3, exc)

                    # Stage 3: Generate English track if needed and audio wasn't English
                    if "eng" in target_langs and "eng" not in generated_srts:
                        logger.info("  Generating Whisper English translation track...")
                        eng_res = transcriber.transcribe(
                            wav_path, model, language=detected_short, task="translate", check_idle=not ignore_idle
                        )
                        if eng_res.aborted:
                            logger.warning("English translation aborted for %s — user active", display)
                            aborted = True
                            break
                        if eng_res.success and eng_res.srt_text:
                            generated_srts["eng"] = eng_res.srt_text
                            try:
                                jellyfin_client.upload_subtitle(item_id, eng_res.srt_text, language="eng")
                                logger.info("  Uploaded Whisper English subtitle for %s (%d segments)",
                                            display, eng_res.segment_count)
                                successful_langs.append("eng")
                                jellyfin_client.mark_processed(
                                    item_id=item_id,
                                    status="in_progress",
                                    name=display,
                                    item_type=item_type,
                                    completed_langs=["eng"],
                                    detail=f"Whisper English: {eng_res.segment_count} segments",
                                )
                            except Exception as exc:
                                logger.error("Upload failed for %s (eng): %s", display, exc)

                    if aborted:
                        break

                    # Stage 4: Translate to Romanian and other target languages locally via NLLB-200
                    base_srt_lang = "eng" if "eng" in generated_srts else detected_3
                    base_srt = generated_srts.get(base_srt_lang)

                    for tgt in target_langs:
                        if tgt in successful_langs:
                            continue  # already uploaded

                        if base_srt:
                            logger.info("  [3/3] Translating subtitle (%s -> %s) locally using NLLB-200...",
                                        base_srt_lang, tgt)
                            trans_res = translator.translate_srt(
                                base_srt, src_lang=base_srt_lang, tgt_lang=tgt, check_idle=not ignore_idle
                            )
                            if trans_res.aborted:
                                logger.warning("Translation aborted for %s (%s) — user active", display, tgt)
                                aborted = True
                                break
                            if trans_res.success and trans_res.srt_text:
                                generated_srts[tgt] = trans_res.srt_text
                                try:
                                    jellyfin_client.upload_subtitle(item_id, trans_res.srt_text, language=tgt)
                                    logger.info("  Uploaded translated %s subtitle for %s (%d blocks)",
                                                tgt, display, trans_res.line_count)
                                    successful_langs.append(tgt)
                                    jellyfin_client.mark_processed(
                                        item_id=item_id,
                                        status="in_progress",
                                        name=display,
                                        item_type=item_type,
                                        completed_langs=[tgt],
                                        detail=f"NLLB {tgt}: {trans_res.line_count} blocks",
                                    )
                                except Exception as exc:
                                    logger.error("Upload failed for %s (%s): %s", display, tgt, exc)

                    if aborted:
                        break

                    # Check overall completion
                    pending_still = jellyfin_client.get_item_pending_languages(item_id)
                    if not pending_still:
                        jellyfin_client.mark_processed(
                            item_id=item_id,
                            status="success",
                            name=display,
                            item_type=item_type,
                            detail=f"All {len(config.SUBTITLE_LANGS)} tracks complete",
                        )
                    elif not successful_langs:
                        jellyfin_client.mark_processed(
                            item_id=item_id,
                            status="failed",
                            name=display,
                            item_type=item_type,
                            detail="Failed to generate any target subtitle tracks",
                        )

                finally:
                    transcriber._safe_delete(wav_path)

                if aborted:
                    break

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

