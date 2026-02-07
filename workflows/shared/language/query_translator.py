"""
Haiku-powered query translation for multilingual search.

Translates search queries to target languages using Claude Haiku for
fast, cost-effective translation. Routes through unified invoke() for
automatic broker routing and cost optimization.

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

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import ModelTier, invoke, invoke_batch, InvokeConfig

logger = logging.getLogger(__name__)

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
        logger.debug(f"Translated query to {target_language_code}: {query[:30]}... -> {result[:30]}...")
        return result


async def _do_query_translation(query: str, target_language: str) -> str:
    """Perform the actual query translation using Haiku via invoke()."""
    user_prompt = f"Translate to {target_language}: {query}"

    try:
        response = await invoke(
            tier=ModelTier.HAIKU,
            system=QUERY_TRANSLATION_SYSTEM,
            user=user_prompt,
            config=InvokeConfig(
                batch_policy=BatchPolicy.PREFER_BALANCE,
                max_tokens=256,
            ),
        )
        return response.content.strip()

    except Exception as e:
        logger.warning(f"Query translation failed: {e}, using original query")
        return query


async def translate_queries(
    queries: list[str],
    target_language_code: str,
    target_language_name: str,
) -> list[str]:
    """Translate multiple search queries.

    Routes through unified invoke() with batching for cost optimization.
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

    logger.debug(f"Submitting {len(uncached_queries)} queries for translation to {target_language_name}")

    # Use invoke_batch for efficient batching
    async with invoke_batch() as batch:
        for query in uncached_queries:
            user_prompt = f"Translate to {target_language_name}: {query}"
            batch.add(
                tier=ModelTier.HAIKU,
                system=QUERY_TRANSLATION_SYSTEM,
                user=user_prompt,
                config=InvokeConfig(
                    batch_policy=BatchPolicy.PREFER_BALANCE,
                    max_tokens=256,
                ),
            )

    # Collect results from batch
    batch_results = await batch.results()
    translated = []
    for i, response in enumerate(batch_results):
        try:
            translated.append(response.content.strip())
        except Exception as e:
            logger.warning(f"Translation failed for query '{uncached_queries[i][:30]}...': {e}, using original")
            translated.append(uncached_queries[i])

    # Merge results and update cache
    for i, idx in enumerate(uncached_indices):
        results[idx] = translated[i]
        cache_key = f"query_{target_language_code}_{hash(uncached_queries[i])}"
        _query_cache[cache_key] = translated[i]

    return results


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
