"""Different search sources for paper retrieval."""

import logging
from typing import Any

from langchain_tools.base import get_store_manager

logger = logging.getLogger(__name__)


async def semantic_search(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Semantic search using embeddings.

    Searches entire corpus without filtering.

    Returns list of dicts with zotero_key, similarity score, and metadata.
    """
    store_manager = get_store_manager()

    try:
        query_embedding = await store_manager.embedding.embed(query)

        # Search Chroma for semantically similar content
        results = await store_manager.chroma.search(
            query_embedding=query_embedding,
            n_results=limit * 2,  # Get extra to allow for deduplication
            where=None,  # No filter - search all
        )

        search_results: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for r in results:
            metadata = r.get("metadata", {})
            zotero_key = metadata.get("zotero_key")
            if not zotero_key or zotero_key in seen_keys:
                continue

            seen_keys.add(zotero_key)
            similarity = 1 - r.get("distance", 1)  # Convert distance to similarity

            search_results.append(
                {
                    "zotero_key": zotero_key,
                    "score": similarity,
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", 0),
                    "authors": metadata.get("authors", []),
                }
            )

        return search_results[:limit]

    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []


async def keyword_search(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Keyword search using Elasticsearch text matching.

    Searches entire corpus without filtering.

    Returns list of dicts with zotero_key, relevance score, and metadata.
    """
    store_manager = get_store_manager()

    try:
        es_query = {"match": {"content": query}}
        records = await store_manager.es_stores.store.search(
            query=es_query,
            size=limit * 2,
        )

        search_results: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for r in records:
            if not r.zotero_key or r.zotero_key in seen_keys:
                continue

            seen_keys.add(r.zotero_key)
            # Normalize score to 0-1 range (ES scores vary widely)
            score = min(1.0, getattr(r, "_score", 1.0) / 10.0)

            # Get metadata from record
            metadata = r.metadata or {}
            search_results.append(
                {
                    "zotero_key": r.zotero_key,
                    "score": score,
                    "title": metadata.get("title", "Unknown"),
                    "year": metadata.get("year", 0),
                    "authors": metadata.get("authors", []),
                }
            )

        return search_results[:limit]

    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []


def merge_search_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Merge and rank results from both search methods.

    Uses reciprocal rank fusion for fair combination.
    """
    # Build score map and metadata storage
    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    k = 60  # RRF constant

    # Add semantic scores using RRF
    for rank, result in enumerate(semantic_results):
        key = result["zotero_key"]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in metadata:
            metadata[key] = result

    # Add keyword scores using RRF
    for rank, result in enumerate(keyword_results):
        key = result["zotero_key"]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in metadata:
            metadata[key] = result

    # Sort by combined score and build output
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    merged_results: list[dict[str, Any]] = []
    for key in sorted_keys[:limit]:
        result = metadata[key].copy()
        result["score"] = scores[key]
        merged_results.append(result)

    return merged_results
