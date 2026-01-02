"""Re-export from shared module for backward compatibility."""

from workflows.shared.language import (
    LanguageConfig,
    LANGUAGE_NAMES,
    LANGUAGE_DOMAINS,
    LANGUAGE_LOCALES,
    get_language_config,
    is_supported_language,
    get_all_supported_languages,
)

__all__ = [
    "LanguageConfig",
    "LANGUAGE_NAMES",
    "LANGUAGE_DOMAINS",
    "LANGUAGE_LOCALES",
    "get_language_config",
    "is_supported_language",
    "get_all_supported_languages",
]
