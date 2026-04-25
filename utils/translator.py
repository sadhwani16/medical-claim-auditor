"""Language detection and translation using Sarvam AI (Indian model) with fallbacks."""

import requests
from langdetect import detect, LangDetectException
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SARVAM_API_KEY, SARVAM_LANG_MAP, SUPPORTED_LANGUAGES


def detect_language(text: str) -> str:
    """Return ISO 639-1 language code (e.g., 'hi', 'ta', 'en')."""
    try:
        lang = detect(text[:500])
        return lang
    except LangDetectException:
        return "en"


def translate_to_english(text: str, source_lang: str = None) -> dict:
    """
    Translate text to English.

    Returns:
        {
            "original_text": str,
            "translated_text": str,
            "source_language": str,
            "source_language_name": str,
            "method": str,
            "translation_needed": bool,
        }
    """
    if source_lang is None:
        source_lang = detect_language(text)

    lang_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang.upper())
    result = {
        "original_text": text,
        "translated_text": text,
        "source_language": source_lang,
        "source_language_name": lang_name,
        "method": "none",
        "translation_needed": source_lang != "en",
    }

    if source_lang == "en":
        return result

    # Try Sarvam AI first (Indian model — preferred)
    if SARVAM_API_KEY:
        translated = _sarvam_translate(text, source_lang)
        if translated:
            result["translated_text"] = translated
            result["method"] = "sarvam-ai"
            return result

    # Fallback: deep-translator (Google Translate wrapper)
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        result["translated_text"] = translated or text
        result["method"] = "google-translate-fallback"
    except Exception:
        result["method"] = "none (translation failed)"

    return result


def _sarvam_translate(text: str, source_lang: str) -> str | None:
    """Call Sarvam AI /translate endpoint."""
    sarvam_code = SARVAM_LANG_MAP.get(source_lang)
    if not sarvam_code:
        return None

    # Sarvam handles max ~1000 chars per request — chunk if needed
    if len(text) > 900:
        chunks = _chunk_text(text, 900)
        translated_chunks = [_sarvam_translate(c, source_lang) for c in chunks]
        if all(t is not None for t in translated_chunks):
            return " ".join(translated_chunks)
        return None

    try:
        resp = requests.post(
            "https://api.sarvam.ai/translate",
            headers={"api-subscription-key": SARVAM_API_KEY},
            json={
                "input": text,
                "source_language_code": sarvam_code,
                "target_language_code": "en-IN",
                "speaker_gender": "Male",
                "mode": "formal",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("translated_text")
    except requests.RequestException:
        pass
    return None


def _chunk_text(text: str, size: int) -> list[str]:
    words = text.split()
    chunks, current = [], []
    count = 0
    for word in words:
        current.append(word)
        count += len(word) + 1
        if count >= size:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks
