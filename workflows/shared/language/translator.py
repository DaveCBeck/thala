"""
Opus-powered prompt translation with caching.

Translates English prompts to target languages using Claude Opus for
high-quality, context-aware translation that preserves prompt intent.

This is NOT machine translation - Opus understands the semantic purpose
of prompts and produces natural, fluent translations.

Usage:
    translated = await translate_prompt(
        CLARIFY_INTENT_SYSTEM,
        target_language="Spanish",
        cache_key="clarify_intent_system_es",
    )
"""

import asyncio
import logging
from typing import Optional

from cachetools import TTLCache

from workflows.shared.llm_utils import get_llm, ModelTier
from workflows.shared.retry_utils import with_retry
from workflows.shared.llm_utils.response_parsing import extract_response_content

logger = logging.getLogger(__name__)

_prompt_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)
_translation_locks: dict[str, asyncio.Lock] = {}


PROMPT_TRANSLATION_SYSTEM = """You are translating an LLM system prompt to another language.

Your task is to produce a high-quality, native-sounding translation that:
1. Preserves the EXACT meaning and instructional intent of the original
2. Uses natural, fluent phrasing in the target language (not literal word-for-word translation)
3. Maintains the same professional, clear, direct tone
4. Keeps the same structure and formatting (sections, bullet points, etc.)

CRITICAL RULES:
- Keep all format placeholders EXACTLY as-is: {date}, {query}, {research_brief}, etc.
- Keep JSON schema examples in English - LLMs understand English JSON schemas universally
- Keep technical terms that are commonly used in English (e.g., "JSON", "API", "URL")
- Keep code examples and variable names in English
- Translate instructional text, section headers, and natural language content

Output ONLY the translated prompt text. No explanations, no "Here's the translation:", just the translated prompt."""


async def translate_prompt(
    english_prompt: str,
    target_language: str,
    cache_key: Optional[str] = None,
) -> str:
    """Translate an English prompt to the target language using Opus."""
    if cache_key and cache_key in _prompt_cache:
        logger.debug(f"Prompt translation cache hit: {cache_key}")
        return _prompt_cache[cache_key]

    if cache_key:
        if cache_key not in _translation_locks:
            _translation_locks[cache_key] = asyncio.Lock()

        async with _translation_locks[cache_key]:
            if cache_key in _prompt_cache:
                return _prompt_cache[cache_key]

            result = await _do_translation(english_prompt, target_language)

            _prompt_cache[cache_key] = result
            logger.debug(f"Translated and cached prompt: {cache_key} ({len(result)} chars)")
            return result
    else:
        return await _do_translation(english_prompt, target_language)


async def _do_translation(english_prompt: str, target_language: str) -> str:
    """Perform the actual translation using Opus."""
    llm = get_llm(ModelTier.OPUS, max_tokens=8192)

    user_prompt = f"""Translate this LLM prompt to {target_language}:

{english_prompt}"""

    async def _invoke():
        response = await llm.ainvoke([
            {"role": "system", "content": PROMPT_TRANSLATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])
        return extract_response_content(response)

    try:
        return await with_retry(_invoke, max_attempts=2)
    except Exception as e:
        logger.error(f"Translation failed after retries: {e}")
        logger.warning("Falling back to English prompt")
        return english_prompt


async def get_translated_prompt(
    english_prompt: str,
    language_code: str,
    language_name: str,
    prompt_name: str,
) -> str:
    """Convenience function to get a translated prompt with standard cache key."""
    if language_code == "en":
        return english_prompt

    cache_key = f"{prompt_name}_{language_code}"
    return await translate_prompt(
        english_prompt,
        target_language=language_name,
        cache_key=cache_key,
    )


def clear_translation_cache() -> None:
    """Clear the prompt translation cache."""
    _prompt_cache.clear()
    logger.debug("Prompt translation cache cleared")


def get_cache_stats() -> dict:
    """Get cache statistics for debugging."""
    return {
        "size": len(_prompt_cache),
        "maxsize": _prompt_cache.maxsize,
        "ttl": _prompt_cache.ttl,
        "keys": list(_prompt_cache.keys()),
    }
