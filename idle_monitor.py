"""
idle_monitor.py — Windows user-input and NVIDIA GPU load detection.

Uses Win32 API (ctypes) for keyboard/mouse idle time and nvidia-smi for GPU
utilization.  All subprocess calls use CREATE_NO_WINDOW so nothing pops up.
"""

import ctypes
import logging
import subprocess
import sys

import config

logger = logging.getLogger(__name__)

# ── Platform Detection & Win32 Structures ─────────────────────────────────────
_IS_WINDOWS = sys.platform == "win32"
_CREATE_NO_WINDOW = 0x08000000 if _IS_WINDOWS else 0

if _IS_WINDOWS:
    import ctypes.wintypes

    class _LASTINPUTINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.UINT),
            ("dwTime", ctypes.wintypes.DWORD),
        ]


# ── Public API ───────────────────────────────────────────────────────────────

def get_idle_duration_seconds() -> float:
    """Return the number of seconds since the last keyboard/mouse input."""
    if not _IS_WINDOWS:
        # Non-Windows systems (e.g. Linux servers / Docker): treat as dedicated idle environment
        return float("inf")

    lii = _LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        logger.warning("GetLastInputInfo failed — assuming 0 idle time")
        return 0.0
    millis_now = ctypes.windll.kernel32.GetTickCount()
    # GetTickCount wraps at ~49.7 days; handle wrap-around
    idle_ms = (millis_now - lii.dwTime) & 0xFFFFFFFF
    return idle_ms / 1000.0


def is_gpu_busy(threshold_percent: int = 15) -> bool:
    """
    Return True if the NVIDIA GPU utilization exceeds *threshold_percent*.

    Falls back to False (GPU assumed free) if nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        # nvidia-smi may list multiple GPUs — take the max utilization
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            logger.debug("nvidia-smi returned no output — assuming GPU is free")
            return False
        max_util = max(int(l) for l in lines)
        logger.debug("GPU utilization: %d%% (threshold %d%%)", max_util, threshold_percent)
        return max_util > threshold_percent
    except FileNotFoundError:
        logger.warning("nvidia-smi not found on PATH — assuming GPU is free")
        return False
    except Exception as exc:
        logger.warning("nvidia-smi query failed (%s) — assuming GPU is free", exc)
        return False


def get_gpu_utilization() -> int | None:
    """Return GPU utilization percentage, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None
        return max(int(l) for l in lines)
    except Exception:
        return None


def is_ready_for_work() -> bool:
    """
    Return True only when the user has been idle for at least
    IDLE_THRESHOLD_SECONDS *and* the GPU utilization is below 15%.
    """
    idle = get_idle_duration_seconds()
    if idle < config.IDLE_THRESHOLD_SECONDS:
        logger.debug("User active (idle %.0fs < threshold %ds)",
                      idle, config.IDLE_THRESHOLD_SECONDS)
        return False
    if is_gpu_busy():
        logger.debug("GPU is busy — deferring work")
        return False
    logger.info("System is idle (%.0fs) and GPU is free — ready for work", idle)
    return True

