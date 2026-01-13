"""Re-export from shared module for backward compatibility."""

from workflows.shared.language import (
    translate_prompt,
    get_translated_prompt,
    clear_translation_cache,
    get_cache_stats,
    PROMPT_TRANSLATION_SYSTEM,
)

__all__ = [
    "translate_prompt",
    "get_translated_prompt",
    "clear_translation_cache",
    "get_cache_stats",
    "PROMPT_TRANSLATION_SYSTEM",
]
