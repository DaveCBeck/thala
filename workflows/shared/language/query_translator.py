"""
Haiku-powered query translation for multilingual search.

Translates search queries to target languages using Claude Haiku for
fast, cost-effective translation. Uses Anthropic Batch API for 50%
cost reduction when translating 5+ queries.

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
import os
from typing import Optional

from cachetools import TTLCache

from workflows.shared.llm_utils import get_llm, ModelTier
from workflows.shared.llm_utils.response_parsing import extract_response_content
from workflows.shared.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

_USE_BATCH_API = os.getenv("THALA_PREFER_BATCH_API", "").lower() in ("true", "1", "yes")

_query_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)
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
    """Translate a search query to the target language using Haiku."""
    if target_language_code == "en":
        return query

    if cache_key is None:
        cache_key = f"query_{target_language_code}_{hash(query)}"

    if cache_key in _query_cache:
        logger.debug(f"Query translation cache hit: {cache_key[:50]}...")
        return _query_cache[cache_key]

    if cache_key not in _query_locks:
        _query_locks[cache_key] = asyncio.Lock()

    async with _query_locks[cache_key]:
        if cache_key in _query_cache:
            return _query_cache[cache_key]

        result = await _do_query_translation(query, target_language_name)

        _query_cache[cache_key] = result
        logger.debug(
            f"Translated query to {target_language_code}: {query[:30]}... -> {result[:30]}..."
        )
        return result


async def _do_query_translation(query: str, target_language: str) -> str:
    """Perform the actual query translation using Haiku."""
    llm = get_llm(ModelTier.HAIKU, max_tokens=256)

    user_prompt = f"Translate to {target_language}: {query}"

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": QUERY_TRANSLATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
        )

        return extract_response_content(response)

    except Exception as e:
        logger.warning(f"Query translation failed: {e}, using original query")
        return query


async def translate_queries(
    queries: list[str],
    target_language_code: str,
    target_language_name: str,
    max_concurrent: int = 5,  # Kept for API compatibility
) -> list[str]:
    """Translate multiple search queries.

    Uses Anthropic Batch API for 50% cost reduction when translating 5+ queries.
    Falls back to concurrent individual calls for smaller batches.
    """
    if target_language_code == "en":
        return queries

    if not queries:
        return []

    # Check cache first, identify queries that need translation
    uncached_queries = []
    uncached_indices = []
    results = [None] * len(queries)

    for i, query in enumerate(queries):
        cache_key = f"query_{target_language_code}_{hash(query)}"
        if cache_key in _query_cache:
            results[i] = _query_cache[cache_key]
        else:
            uncached_queries.append(query)
            uncached_indices.append(i)

    if not uncached_queries:
        return results

    # Use batch API for 5+ uncached queries (50% cost reduction)
    if _USE_BATCH_API and len(uncached_queries) >= 5:
        translated = await _translate_queries_batched(
            uncached_queries, target_language_code, target_language_name
        )
    else:
        # Fall back to concurrent calls for small batches
        semaphore = asyncio.Semaphore(max_concurrent)

        async def translate_with_semaphore(query: str) -> str:
            async with semaphore:
                return await translate_query(
                    query, target_language_code, target_language_name
                )

        translated = await asyncio.gather(
            *[translate_with_semaphore(q) for q in uncached_queries]
        )

    # Merge results and update cache
    for i, idx in enumerate(uncached_indices):
        results[idx] = translated[i]
        cache_key = f"query_{target_language_code}_{hash(uncached_queries[i])}"
        _query_cache[cache_key] = translated[i]

    return results


async def _translate_queries_batched(
    queries: list[str],
    target_language_code: str,
    target_language_name: str,
) -> list[str]:
    """Translate queries using Anthropic Batch API for 50% cost reduction."""
    processor = BatchProcessor(poll_interval=30)

    for i, query in enumerate(queries):
        user_prompt = f"Translate to {target_language_name}: {query}"
        processor.add_request(
            custom_id=f"translate-{i}",
            prompt=user_prompt,
            model=ModelTier.HAIKU,
            max_tokens=256,
            system=QUERY_TRANSLATION_SYSTEM,
        )

    logger.debug(
        f"Submitting batch of {len(queries)} queries for translation to {target_language_name}"
    )
    results = await processor.execute_batch()

    translated = []
    for i, query in enumerate(queries):
        result = results.get(f"translate-{i}")
        if result and result.success:
            translated.append(result.content.strip())
        else:
            logger.warning(
                f"Translation failed for query '{query[:30]}...', using original"
            )
            translated.append(query)

    return translated


def clear_query_cache() -> None:
    """Clear the query translation cache."""
    _query_cache.clear()
    logger.debug("Query translation cache cleared")


def get_query_cache_stats() -> dict:
    """Get cache statistics for debugging."""
    return {
        "size": len(_query_cache),
        "maxsize": _query_cache.maxsize,
        "ttl": _query_cache.ttl,
    }
