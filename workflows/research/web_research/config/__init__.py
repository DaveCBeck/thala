"""Configuration for deep research workflow."""

from workflows.research.web_research.config.languages import (
    LANGUAGE_NAMES,
    LANGUAGE_DOMAINS,
    LANGUAGE_LOCALES,
    get_language_config,
)

__all__ = [
    "LANGUAGE_NAMES",
    "LANGUAGE_DOMAINS",
    "LANGUAGE_LOCALES",
    "get_language_config",
]
