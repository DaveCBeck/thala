"""Language configuration types for multi-lingual support."""

from typing_extensions import TypedDict


class LanguageConfig(TypedDict):
    """Configuration for a specific language in the research workflow."""

    code: str  # ISO 639-1 code (e.g., "es", "zh", "ja")
    name: str  # Full language name (e.g., "Spanish", "Mandarin Chinese")
    search_domains: list[str]  # Preferred domain TLDs (e.g., [".es", ".mx"])
    search_engine_locale: str  # Locale code for search APIs (e.g., "es-ES")


class TranslationConfig(TypedDict):
    """Configuration for translating the final research output."""

    enabled: bool  # Whether to translate the final report
    target_language: str  # Target language code (e.g., "en")
    preserve_quotes: bool  # Keep direct quotes in original language
    preserve_citations: bool  # Keep citation format unchanged
