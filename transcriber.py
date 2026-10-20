"""
transcriber.py — FFmpeg HTTP audio extraction + interruptible faster-whisper pipeline.

The entire pipeline is designed to abort within seconds if the user touches
the mouse or keyboard (idle time drops below 10 s).
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import shutil
import config
from idle_monitor import get_idle_duration_seconds

logger = logging.getLogger(__name__)

# Fix PyAV >=14 compatibility with faster-whisper (metadata_errors was removed from av.open)
try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception as _e:
    pass

# Subprocess flag to suppress console windows on Windows
_CREATE_NO_WINDOW = 0x08000000

# If idle time drops below this many seconds, abort immediately
_ACTIVITY_ABORT_THRESHOLD = 10


def get_ffmpeg_binary() -> str:
    """Find the ffmpeg executable on PATH or in standard installation directories."""
    custom = os.getenv("FFMPEG_PATH")
    if custom and Path(custom).is_file():
        return custom
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        r"C:\Program Files\Jellyfin\Server\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    return "ffmpeg"


# ── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    success: bool
    srt_text: str = ""
    aborted: bool = False
    error: str = ""
    segment_count: int = 0
    detected_language: str = ""
    language_probability: float = 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: list[tuple[float, float, str]], watermark: bool = True) -> str:
    """
    Convert a list of (start, end, text) tuples into an SRT string.
    Optionally prepends an opening badge '[Subtitles: Whisper AI]' if the first
    spoken line doesn't start in the first 2 seconds.
    """
    lines: list[str] = []
    idx = 1

    if watermark and segments:
        first_start = segments[0][0]
        if first_start >= 2.0:
            end_time = min(first_start - 0.3, 3.5)
            lines.append(str(idx))
            lines.append(f"00:00:00,500 --> {_format_timestamp(end_time)}")
            lines.append("[Subtitles: Whisper AI]")
            lines.append("")
            idx += 1

    for start, end, text in segments:
        lines.append(str(idx))
        lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")  # blank line separator
        idx += 1

    return "\n".join(lines)


def _user_became_active() -> bool:
    """Return True if the user has recently interacted with the machine."""
    idle = get_idle_duration_seconds()
    if idle < _ACTIVITY_ABORT_THRESHOLD:
        logger.info("User activity detected (idle %.1fs) — aborting", idle)
        return True
    return False


# ── FFmpeg Audio Extraction ──────────────────────────────────────────────────

def extract_audio(stream_url: str, output_path: str, check_idle: bool = True) -> bool:
    """
    Stream media from *stream_url* via FFmpeg and extract a 16 kHz mono WAV.

    Returns True on success, False if aborted due to user activity or error.
    The caller is responsible for cleaning up *output_path* on abort.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [ffmpeg_bin, "-y"]
    if stream_url.startswith("http"):
        cmd.extend([
            "-headers",
            f'Authorization: MediaBrowser Token="{config.JELLYFIN_API_KEY}"\r\n',
        ])
    cmd.extend([
        "-i", stream_url,
        "-vn",                    # drop video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16 kHz
        "-ac", "1",               # mono
        output_path,
    ])
    logger.info("Starting FFmpeg (%s): extracting audio to %s", ffmpeg_bin, output_path)

    try:
        with tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                creationflags=_CREATE_NO_WINDOW,
            )

            # Poll the process and check for user activity
            import time
            while proc.poll() is None:
                if check_idle and _user_became_active():
                    logger.warning("Aborting FFmpeg extraction due to user activity")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    # Clean up partial file
                    _safe_delete(output_path)
                    return False
                time.sleep(0.5)

            if proc.returncode != 0:
                stderr_file.seek(0)
                stderr_text = stderr_file.read().decode("utf-8", errors="replace")
                logger.error("FFmpeg exited with code %d: %s", proc.returncode,
                              stderr_text[:500])
                _safe_delete(output_path)
                return False
    except FileNotFoundError:
        logger.error("ffmpeg not found — cannot extract audio. Checked PATH and %s", ffmpeg_bin)
        return False

    logger.info("FFmpeg extraction complete: %s", output_path)
    return True


# ── Whisper Transcription ────────────────────────────────────────────────────

def transcribe(audio_path: str, model: object,
               language: str | None = None,
               task: str = "transcribe",
               multilingual: bool = True,
               check_idle: bool = True) -> TranscriptionResult:
    """
    Transcribe *audio_path* using a pre-loaded faster-whisper model.

    Args:
        audio_path: Path to the 16 kHz mono WAV file.
        model: A loaded faster_whisper.WhisperModel instance.
        language: Force source language (e.g. "ro", "en"). None = auto-detect.
        task: "transcribe" for native-language text, "translate" for English translation.
        multilingual: If True, dynamically re-detect language per segment/scene (e.g. Spanish scenes in English series).
        check_idle: If True, abort when user activity is detected.

    Checks for user activity after every segment — if the user touches
    the keyboard or mouse, transcription aborts immediately.

    Returns:
        A TranscriptionResult with the SRT text or abort/error info.
    """
    logger.info("Starting Whisper %s of %s (language=%s, multilingual=%s)",
                task, audio_path, language or "auto", multilingual)

    try:
        segments_gen, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            multilingual=multilingual,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            repetition_penalty=1.1,
        )
        logger.info("Detected language: %s (prob %.2f)", info.language, info.language_probability)
    except Exception as exc:
        logger.error("Whisper transcribe() failed: %s", exc)
        return TranscriptionResult(success=False, error=str(exc))

    collected: list[tuple[float, float, str]] = []

    try:
        for segment in segments_gen:
            # ── Activity check on every segment ──
            if check_idle and _user_became_active():
                logger.warning("Aborting transcription at segment %d due to user activity",
                               len(collected) + 1)
                _safe_delete(audio_path)
                return TranscriptionResult(success=False, aborted=True,
                                           segment_count=len(collected))

            text = segment.text.strip()
            if text:
                collected.append((segment.start, segment.end, text))
                logger.debug("[%s --> %s] %s",
                             _format_timestamp(segment.start),
                             _format_timestamp(segment.end), text)
    except Exception as exc:
        logger.error("Error during segment iteration: %s", exc)
        return TranscriptionResult(success=False, error=str(exc))

    if not collected:
        logger.warning("No speech segments detected — file may be instrumental or silent")
        return TranscriptionResult(
            success=True, srt_text="", segment_count=0,
            detected_language=info.language,
            language_probability=info.language_probability,
        )

    srt_text = _segments_to_srt(collected)
    logger.info("Transcription complete: %d segments (language=%s)", len(collected), info.language)
    return TranscriptionResult(
        success=True, srt_text=srt_text,
        segment_count=len(collected),
        detected_language=info.language,
        language_probability=info.language_probability,
    )


# ── Full Pipeline ────────────────────────────────────────────────────────────

def process_item(stream_url: str, model: object,
                 language: str | None = None,
                 task: str = "transcribe",
                 multilingual: bool = True,
                 check_idle: bool = True) -> TranscriptionResult:
    """
    End-to-end pipeline: download audio via FFmpeg → transcribe with Whisper.

    Handles temp file lifecycle and activity-based abort at every stage.
    """
    # Create a temp WAV file in the project directory
    tmp_dir = Path(__file__).parent
    fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
    os.close(fd)

    try:
        # Stage 1: FFmpeg extraction
        if not extract_audio(stream_url, wav_path, check_idle=check_idle):
            return TranscriptionResult(success=False, aborted=True)

        # Stage 2: Whisper transcription
        result = transcribe(
            wav_path, model, language=language, task=task,
            multilingual=multilingual, check_idle=check_idle,
        )
        return result

    finally:
        # Always clean up the temp WAV file
        _safe_delete(wav_path)


def _safe_delete(path: str) -> None:
    """Delete a file if it exists, suppressing any errors."""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
            logger.debug("Deleted temp file: %s", path)
    except OSError as exc:
        logger.warning("Could not delete %s: %s", path, exc)

