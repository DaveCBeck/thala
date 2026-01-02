"""Language configuration types for multi-lingual support."""

from typing_extensions import TypedDict


class LanguageConfig(TypedDict):
    """Configuration for a specific language in research workflows.

    Attributes:
        code: ISO 639-1 code (e.g., "es", "zh", "ja")
        name: Full language name (e.g., "Spanish", "Mandarin Chinese")
        search_domains: Preferred domain TLDs (e.g., [".es", ".mx"])
        search_engine_locale: Locale code for search APIs (e.g., "es-ES")
    """

    code: str
    name: str
    search_domains: list[str]
    search_engine_locale: str


class TranslationConfig(TypedDict):
    """Configuration for translating final output.

    Attributes:
        enabled: Whether to translate the final report
        target_language: Target language code (e.g., "en")
        preserve_quotes: Keep direct quotes in original language
        preserve_citations: Keep citation format unchanged
    """

    enabled: bool
    target_language: str
    preserve_quotes: bool
    preserve_citations: bool
