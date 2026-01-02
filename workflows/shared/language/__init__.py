"""
Shared language utilities for multi-lingual workflow support.

Provides:
- Language configuration types and data for 30 supported languages
- Opus-powered prompt translation with caching
- Haiku-powered query translation for search

Example usage:
    from workflows.shared.language import (
        LanguageConfig,
        get_language_config,
        get_translated_prompt,
        translate_query,
    )

    # Get language config
    config = get_language_config("es")  # Spanish

    # Translate a prompt (uses Opus, cached 24h)
    translated_prompt = await get_translated_prompt(
        SYSTEM_PROMPT,
        language_code="es",
        language_name="Spanish",
        prompt_name="my_node_system",
    )

    # Translate a search query (uses Haiku, cached 1h)
    translated_query = await translate_query(
        "machine learning healthcare",
        target_language_code="es",
        target_language_name="Spanish",
    )
"""

from .types import LanguageConfig, TranslationConfig
from .languages import (
    LANGUAGE_NAMES,
    LANGUAGE_DOMAINS,
    LANGUAGE_LOCALES,
    get_language_config,
    is_supported_language,
    get_all_supported_languages,
)
from .translator import (
    translate_prompt,
    get_translated_prompt,
    clear_translation_cache,
    get_cache_stats,
    PROMPT_TRANSLATION_SYSTEM,
)
from .query_translator import (
    translate_query,
    translate_queries,
    clear_query_cache,
    get_query_cache_stats,
)

__all__ = [
    # Types
    "LanguageConfig",
    "TranslationConfig",
    # Language data
    "LANGUAGE_NAMES",
    "LANGUAGE_DOMAINS",
    "LANGUAGE_LOCALES",
    # Language helpers
    "get_language_config",
    "is_supported_language",
    "get_all_supported_languages",
    # Prompt translation
    "translate_prompt",
    "get_translated_prompt",
    "clear_translation_cache",
    "get_cache_stats",
    "PROMPT_TRANSLATION_SYSTEM",
    # Query translation
    "translate_query",
    "translate_queries",
    "clear_query_cache",
    "get_query_cache_stats",
]
