"""
config.py — Loads and validates environment variables from .env
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


def _require(name: str) -> str:
    """Return an env var or exit with an error if it is missing."""
    value = os.getenv(name)
    if not value:
        logger.error("Required environment variable %s is not set. "
                      "Copy .env.example to .env and fill in the values.", name)
        sys.exit(1)
    return value


# ── Required ──────────────────────────────────────────────────────────────────
JELLYFIN_URL: str = _require("JELLYFIN_URL").rstrip("/")
JELLYFIN_API_KEY: str = _require("JELLYFIN_API_KEY")

# ── Optional with defaults ───────────────────────────────────────────────────
IDLE_THRESHOLD_SECONDS: int = int(os.getenv("IDLE_THRESHOLD_SECONDS", "600"))
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "medium")
COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")
SUBTITLE_LANGS: list[str] = [
    s.strip() for s in os.getenv("SUBTITLE_LANGS", "eng").split(",") if s.strip()
]

# Validate
_VALID_MODELS = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
if WHISPER_MODEL not in _VALID_MODELS:
    logger.warning("WHISPER_MODEL='%s' is not in the known set %s. "
                    "Proceeding anyway.", WHISPER_MODEL, _VALID_MODELS)

_VALID_COMPUTE = {"float16", "int8", "int8_float16", "float32"}
if COMPUTE_TYPE not in _VALID_COMPUTE:
    logger.warning("COMPUTE_TYPE='%s' is not in the known set %s. "
                    "Proceeding anyway.", COMPUTE_TYPE, _VALID_COMPUTE)

if IDLE_THRESHOLD_SECONDS < 30:
    logger.warning("IDLE_THRESHOLD_SECONDS=%d is very low; you may get "
                    "frequent interruptions.", IDLE_THRESHOLD_SECONDS)


def summary() -> str:
    """Return a human-readable config summary (redacts the API key)."""
    if len(JELLYFIN_API_KEY) > 8:
        redacted_key = JELLYFIN_API_KEY[:4] + "****" + JELLYFIN_API_KEY[-4:]
    else:
        redacted_key = "****"
    return (
        f"  Jellyfin URL       : {JELLYFIN_URL}\n"
        f"  API Key            : {redacted_key}\n"
        f"  Idle Threshold     : {IDLE_THRESHOLD_SECONDS}s\n"
        f"  Whisper Model      : {WHISPER_MODEL}\n"
        f"  Compute Type       : {COMPUTE_TYPE}\n"
        f"  Subtitle Languages : {', '.join(SUBTITLE_LANGS)}"
    )

