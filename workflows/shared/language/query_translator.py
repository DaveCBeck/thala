"""
Haiku-powered query translation for multilingual search.

Translates search queries to target languages using Claude Haiku for
fast, cost-effective translation. Queries are typically short and don't
require the sophistication of prompt translation.

Usage:
    translated = await translate_query(
        "machine learning applications in healthcare",
        target_language_code="es",
        target_language_name="Spanish",
    )
    # Returns: "aplicaciones de aprendizaje automático en salud"
"""

import asyncio
import logging
from typing import Optional

from cachetools import TTLCache

from workflows.shared.llm_utils import get_llm, ModelTier

logger = logging.getLogger(__name__)

# Cache translated queries (1h TTL - queries are more ephemeral than prompts)
_query_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)

# Lock to prevent concurrent translations of the same query
_query_locks: dict[str, asyncio.Lock] = {}


QUERY_TRANSLATION_SYSTEM = """You are a search query translator. Translate the given search query to the target language.

Rules:
1. Produce a natural, native-sounding query in the target language
2. Preserve the search intent and key concepts
3. Keep technical terms that are commonly used in English if appropriate for the target language
4. Output ONLY the translated query - no explanations, quotes, or prefixes"""


async def translate_query(
    query: str,
    target_language_code: str,
    target_language_name: str,
    cache_key: Optional[str] = None,
) -> str:
    """
    Translate a search query to the target language using Haiku.

    Args:
        query: The English search query to translate
        target_language_code: ISO 639-1 code (e.g., "es", "zh")
        target_language_name: Full language name (e.g., "Spanish")
        cache_key: Optional cache key. If not provided, uses hash of query + language.

    Returns:
        Translated query in the target language
    """
    if target_language_code == "en":
        return query

    # Generate cache key if not provided
    if cache_key is None:
        cache_key = f"query_{target_language_code}_{hash(query)}"

    # Check cache first
    if cache_key in _query_cache:
        logger.debug(f"Query translation cache hit: {cache_key[:50]}...")
        return _query_cache[cache_key]

    # Use lock to prevent concurrent translations of the same query
    if cache_key not in _query_locks:
        _query_locks[cache_key] = asyncio.Lock()

    async with _query_locks[cache_key]:
        # Double-check cache after acquiring lock
        if cache_key in _query_cache:
            return _query_cache[cache_key]

        result = await _do_query_translation(query, target_language_name)

        _query_cache[cache_key] = result
        logger.debug(f"Translated query to {target_language_code}: {query[:30]}... -> {result[:30]}...")
        return result


async def _do_query_translation(query: str, target_language: str) -> str:
    """Perform the actual query translation using Haiku."""
    llm = get_llm(ModelTier.HAIKU, max_tokens=256)

    user_prompt = f"Translate to {target_language}: {query}"

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": QUERY_TRANSLATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])

        # Extract text content
        if isinstance(response.content, str):
            return response.content.strip()
        elif isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "").strip()
                elif hasattr(block, "type") and block.type == "text":
                    return getattr(block, "text", "").strip()

        return str(response.content).strip()

    except Exception as e:
        logger.warning(f"Query translation failed: {e}, using original query")
        return query


async def translate_queries(
    queries: list[str],
    target_language_code: str,
    target_language_name: str,
    max_concurrent: int = 5,
) -> list[str]:
    """
    Translate multiple search queries concurrently.

    Args:
        queries: List of English search queries
        target_language_code: ISO 639-1 code (e.g., "es", "zh")
        target_language_name: Full language name (e.g., "Spanish")
        max_concurrent: Maximum concurrent translation requests

    Returns:
        List of translated queries in the same order
    """
    if target_language_code == "en":
        return queries

    semaphore = asyncio.Semaphore(max_concurrent)

    async def translate_with_semaphore(query: str) -> str:
        async with semaphore:
            return await translate_query(query, target_language_code, target_language_name)

    return await asyncio.gather(*[translate_with_semaphore(q) for q in queries])


def clear_query_cache() -> None:
    """Clear the query translation cache."""
    _query_cache.clear()
    logger.info("Query translation cache cleared")


def get_query_cache_stats() -> dict:
    """Get cache statistics for debugging."""
    return {
        "size": len(_query_cache),
        "maxsize": _query_cache.maxsize,
        "ttl": _query_cache.ttl,
    }
