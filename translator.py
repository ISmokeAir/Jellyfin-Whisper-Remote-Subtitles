"""
translator.py — High-quality local subtitle translation using Meta NLLB-200 & CTranslate2.

Runs 100% locally on CUDA (or CPU) without external APIs or PyTorch overhead.
Integrates idle monitoring so translation aborts if user touches mouse or keyboard.
"""

import gc
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable

import config
from idle_monitor import get_idle_duration_seconds

logger = logging.getLogger(__name__)

# Activity threshold: abort if idle duration is below this
_ACTIVITY_ABORT_THRESHOLD = 10

# Default HuggingFace repo for NLLB-200 distilled INT8 CTranslate2 model
NLLB_MODEL_REPO = "mijuanlo/nllb-200-distilled-600M-ct2-int8"

# ISO-639-1 / ISO-639-2 to NLLB-200 language code mapping
_LANG_TO_NLLB = {
    # Romanian
    "ro": "ron_Latn", "ron": "ron_Latn", "rum": "ron_Latn",
    # English
    "en": "eng_Latn", "eng": "eng_Latn",
    # Spanish
    "es": "spa_Latn", "spa": "spa_Latn",
    # French
    "fr": "fra_Latn", "fra": "fra_Latn", "fre": "fra_Latn",
    # German
    "de": "deu_Latn", "deu": "deu_Latn", "ger": "deu_Latn",
    # Italian
    "it": "ita_Latn", "ita": "ita_Latn",
    # Portuguese
    "pt": "por_Latn", "por": "por_Latn",
    # Russian
    "ru": "rus_Cyrl", "rus": "rus_Cyrl",
    # Ukrainian
    "uk": "ukr_Cyrl", "ukr": "ukr_Cyrl",
    # Polish
    "pl": "pol_Latn", "pol": "pol_Latn",
    # Dutch
    "nl": "nld_Latn", "nld": "nld_Latn", "dut": "nld_Latn",
    # Turkish
    "tr": "tur_Latn", "tur": "tur_Latn",
    # Hungarian
    "hu": "hun_Latn", "hun": "hun_Latn",
    # Japanese
    "ja": "jpn_Jpan", "jpn": "jpn_Jpan",
    # Korean
    "ko": "kor_Hang", "kor": "kor_Hang",
    # Chinese (Simplified)
    "zh": "zho_Hans", "zho": "zho_Hans", "chi": "zho_Hans",
    # Czech
    "cs": "ces_Latn", "ces": "ces_Latn", "cze": "ces_Latn",
    # Greek
    "el": "ell_Grek", "ell": "ell_Grek", "gre": "ell_Grek",
    # Swedish
    "sv": "swe_Latn", "swe": "swe_Latn",
    # Danish
    "da": "dan_Latn", "dan": "dan_Latn",
    # Finnish
    "fi": "fin_Latn", "fin": "fin_Latn",
    # Norwegian
    "no": "nor_Latn", "nor": "nor_Latn",
    # Bulgarian
    "bg": "bul_Cyrl", "bul": "bul_Cyrl",
}

# Localized opening banners
_BANNER_TRANSLATIONS = {
    "ron_Latn": "[Subtitrări: Whisper AI]",
    "spa_Latn": "[Subtítulos: Whisper AI]",
    "fra_Latn": "[Sous-titres : Whisper AI]",
    "deu_Latn": "[Untertitel: Whisper AI]",
    "ita_Latn": "[Sottotitoli: Whisper AI]",
    "por_Latn": "[Legendas: Whisper AI]",
    "rus_Cyrl": "[Субтитры: Whisper AI]",
}


def _setup_cuda_dlls():
    """Ensure CUDA 12 runtime DLLs can be loaded on Windows."""
    if sys.platform != "win32":
        return None
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
            return p
    return None


def _user_became_active() -> bool:
    """Return True if user input was detected recently."""
    return get_idle_duration_seconds() < _ACTIVITY_ABORT_THRESHOLD


@dataclass
class TranslationResult:
    success: bool
    srt_text: str = ""
    aborted: bool = False
    error: str = ""
    line_count: int = 0


class LocalTranslator:
    """Manages the CTranslate2 NLLB-200 model and tokenizer instance."""

    def __init__(self, model_repo: str = NLLB_MODEL_REPO):
        self.model_repo = model_repo
        self.translator = None
        self.tokenizer = None
        self.current_src_lang = None

    def load(self, src_lang_nllb: str = "eng_Latn"):
        """Load NLLB model and tokenizer if not already loaded."""
        if self.translator is not None and self.current_src_lang == src_lang_nllb:
            return

        _setup_cuda_dlls()
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        logger.info("Loading translation model '%s'...", self.model_repo)
        model_path = snapshot_download(self.model_repo)

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        logger.info("Initializing CTranslate2 Translator on device: %s (int8)", device)
        self.translator = ctranslate2.Translator(model_path, device=device, compute_type="int8")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang=src_lang_nllb)
        self.current_src_lang = src_lang_nllb
        logger.info("Translation model ready (source lang: %s)", src_lang_nllb)

    def unload(self):
        """Free translator and tokenizer memory."""
        if self.translator is not None:
            logger.info("Unloading translation model from VRAM...")
            del self.translator
            del self.tokenizer
            self.translator = None
            self.tokenizer = None
            self.current_src_lang = None
            gc.collect()


# Singleton translator instance
_GLOBAL_TRANSLATOR = LocalTranslator()


def get_translator() -> LocalTranslator:
    return _GLOBAL_TRANSLATOR


def unload_translator():
    _GLOBAL_TRANSLATOR.unload()


# ── SRT Parsing & Translation ────────────────────────────────────────────────

_SRT_BLOCK_REGEX = re.compile(
    r"(\d+)\r?\n"
    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3})\r?\n"
    r"([\s\S]*?)(?=\r?\n\r?\n|\Z)",
    re.MULTILINE
)


def translate_srt(
    srt_content: str,
    src_lang: str = "eng",
    tgt_lang: str = "ron",
    batch_size: int = 32,
    check_idle: bool = True,
) -> TranslationResult:
    """
    Translate an SRT subtitle string from *src_lang* to *tgt_lang* locally.

    Args:
        srt_content: The full SRT text to translate.
        src_lang: Source language (e.g. 'eng', 'en').
        tgt_lang: Target language (e.g. 'ron', 'ro', 'spa').
        batch_size: Number of subtitle blocks to translate in parallel on GPU.
        check_idle: If True, check user activity between batches and abort if active.

    Returns:
        TranslationResult with the translated SRT text or error info.
    """
    if not srt_content.strip():
        return TranslationResult(success=True, srt_text="")

    src_nllb = _LANG_TO_NLLB.get(src_lang.lower(), f"{src_lang}_Latn")
    tgt_nllb = _LANG_TO_NLLB.get(tgt_lang.lower(), f"{tgt_lang}_Latn")

    if src_nllb == tgt_nllb:
        # Same language, no translation needed
        return TranslationResult(success=True, srt_text=srt_content)

    logger.info("Translating SRT from %s (%s) to %s (%s)...",
                src_lang, src_nllb, tgt_lang, tgt_nllb)

    # 1. Parse SRT blocks
    blocks = []
    for match in _SRT_BLOCK_REGEX.finditer(srt_content.strip()):
        index_str = match.group(1).strip()
        timestamp_str = match.group(2).strip()
        text_str = match.group(3).strip()
        blocks.append({
            "index": index_str,
            "timestamp": timestamp_str,
            "text": text_str,
        })

    if not blocks:
        logger.warning("No valid SRT blocks found to translate")
        return TranslationResult(success=True, srt_text=srt_content)

    # 2. Load model
    try:
        t = get_translator()
        t.load(src_lang_nllb=src_nllb)
    except Exception as exc:
        logger.error("Failed to load translator model: %s", exc)
        return TranslationResult(success=False, error=str(exc))

    # 3. Separate banner / watermark and dialogue lines to translate
    texts_to_translate: list[str] = []
    translate_map: list[int | None] = []  # maps block_idx -> index in texts_to_translate or None if special

    for idx, b in enumerate(blocks):
        raw_text = b["text"]
        # Handle opening watermark badge directly
        if raw_text.startswith("[Subtitles:") or "Whisper AI" in raw_text:
            localized_banner = _BANNER_TRANSLATIONS.get(tgt_nllb, f"[Subtitles: Whisper AI ({tgt_lang.upper()})]")
            b["translated_text"] = localized_banner
            translate_map.append(None)
        else:
            texts_to_translate.append(raw_text)
            translate_map.append(len(texts_to_translate) - 1)

    # 4. Batch translation with idle checking
    translated_texts: list[str] = [""] * len(texts_to_translate)
    num_texts = len(texts_to_translate)

    try:
        for start_idx in range(0, num_texts, batch_size):
            if check_idle and _user_became_active():
                logger.warning("Aborting translation due to user activity at line %d/%d",
                               start_idx, num_texts)
                return TranslationResult(
                    success=False,
                    aborted=True,
                    line_count=start_idx
                )

            end_idx = min(start_idx + batch_size, num_texts)
            chunk = texts_to_translate[start_idx:end_idx]

            # Tokenize batch
            source_tokens = [
                t.tokenizer.convert_ids_to_tokens(t.tokenizer.encode(txt))
                for txt in chunk
            ]
            target_prefix = [[tgt_nllb]] * len(chunk)

            # Inference
            results = t.translator.translate_batch(
                source_tokens,
                target_prefix=target_prefix,
                beam_size=4,
            )

            for i, res in enumerate(results):
                tokens = res.hypotheses[0][1:]  # skip prefix token
                decoded = t.tokenizer.decode(t.tokenizer.convert_tokens_to_ids(tokens))
                translated_texts[start_idx + i] = decoded.strip()

    except Exception as exc:
        logger.error("Batch translation failed: %s", exc)
        return TranslationResult(success=False, error=str(exc))

    # 5. Assign translated texts back to blocks
    for idx, b in enumerate(blocks):
        t_idx = translate_map[idx]
        if t_idx is not None:
            b["translated_text"] = translated_texts[t_idx]

    # 6. Reconstruct SRT
    output_lines = []
    for b in blocks:
        output_lines.append(b["index"])
        output_lines.append(b["timestamp"])
        output_lines.append(b["translated_text"])
        output_lines.append("")  # blank separator

    translated_srt = "\n".join(output_lines)
    logger.info("Successfully translated %d subtitle blocks to %s", len(blocks), tgt_lang)
    return TranslationResult(
        success=True,
        srt_text=translated_srt,
        line_count=len(blocks)
    )

