"""
Language configuration for multi-lingual research workflow.

Provides language-specific settings for:
- Full language names (for Opus translation prompts)
- Search domain preferences (country-specific TLDs)
- Search engine locales (for Firecrawl/Perplexity)
"""

from typing import Optional
from typing_extensions import TypedDict


class LanguageConfig(TypedDict):
    """Configuration for a specific language."""

    code: str  # ISO 639-1 code
    name: str  # Full language name
    search_domains: list[str]  # Preferred domain TLDs
    search_engine_locale: str  # Locale code for search APIs


# Full language names for Opus translation prompts
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "zh": "Mandarin Chinese",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "pt": "Portuguese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "hi": "Hindi",
    "bn": "Bengali",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
}

# Preferred domain TLDs for language-specific web search
# These are used to boost results from country-specific sites
LANGUAGE_DOMAINS: dict[str, list[str]] = {
    "es": [".es", ".mx", ".ar", ".co", ".cl", ".pe"],
    "zh": [".cn", ".tw", ".hk", ".sg"],
    "ja": [".jp", ".co.jp"],
    "de": [".de", ".at", ".ch"],
    "fr": [".fr", ".ca", ".be", ".ch"],
    "pt": [".br", ".pt"],
    "ko": [".kr", ".co.kr"],
    "ru": [".ru"],
    "ar": [".ae", ".sa", ".eg", ".ma"],
    "it": [".it"],
    "nl": [".nl", ".be"],
    "pl": [".pl"],
    "tr": [".tr"],
    "vi": [".vn"],
    "th": [".th"],
    "id": [".id"],
    "hi": [".in"],
    "bn": [".bd", ".in"],
    "sv": [".se"],
    "no": [".no"],
    "da": [".dk"],
    "fi": [".fi"],
    "cs": [".cz"],
    "el": [".gr"],
    "he": [".il"],
    "uk": [".ua"],
    "ro": [".ro"],
    "hu": [".hu"],
}

# Search engine locale codes for Firecrawl/Perplexity location hints
LANGUAGE_LOCALES: dict[str, str] = {
    "es": "es-ES",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "de": "de-DE",
    "fr": "fr-FR",
    "pt": "pt-BR",
    "ko": "ko-KR",
    "ru": "ru-RU",
    "ar": "ar-SA",
    "it": "it-IT",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "tr": "tr-TR",
    "vi": "vi-VN",
    "th": "th-TH",
    "id": "id-ID",
    "hi": "hi-IN",
    "bn": "bn-BD",
    "sv": "sv-SE",
    "no": "no-NO",
    "da": "da-DK",
    "fi": "fi-FI",
    "cs": "cs-CZ",
    "el": "el-GR",
    "he": "he-IL",
    "uk": "uk-UA",
    "ro": "ro-RO",
    "hu": "hu-HU",
}


def get_language_config(code: str) -> Optional[LanguageConfig]:
    """
    Get full language configuration for a language code.

    Args:
        code: ISO 639-1 language code (e.g., "es", "zh", "ja")

    Returns:
        LanguageConfig with name, domains, and locale, or None if not supported
    """
    if code not in LANGUAGE_NAMES:
        return None

    return LanguageConfig(
        code=code,
        name=LANGUAGE_NAMES[code],
        search_domains=LANGUAGE_DOMAINS.get(code, []),
        search_engine_locale=LANGUAGE_LOCALES.get(code, f"{code}-{code.upper()}"),
    )


def is_supported_language(code: str) -> bool:
    """Check if a language code is supported."""
    return code in LANGUAGE_NAMES


def get_all_supported_languages() -> list[str]:
    """Get list of all supported language codes."""
    return list(LANGUAGE_NAMES.keys())
