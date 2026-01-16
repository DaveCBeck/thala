"""
Paper corpus tools for document enhancement and verification.

Provides hybrid search (semantic + keyword) and content retrieval
for papers in the knowledge store.

Tools:
- search_papers: Find papers by topic using hybrid search
- get_paper_content: Fetch detailed content for a specific paper
"""

import asyncio
import logging
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .base import get_store_manager
from .utils import clamp_limit, output_dict

logger = logging.getLogger(__name__)

# Minimum relevance score for search results
# Papers below this threshold are filtered out to prevent citation drift
# RRF scoring with k=60 gives scores in 0.01-0.03 range for top results
MINIMUM_RELEVANCE_THRESHOLD = 0.008


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------


class PaperResult(BaseModel):
    """Individual paper search result."""

    zotero_key: str = Field(description="Citation key for [@KEY] format")
    title: str = Field(description="Paper title")
    year: int = Field(description="Publication year")
    authors: str = Field(description="Authors (e.g., 'Smith et al.')")
    relevance: float = Field(description="Combined relevance score (0-1)")


class PaperSearchOutput(BaseModel):
    """Output schema for search_papers tool."""

    query: str
    total_found: int
    papers: list[PaperResult]


class PaperContentOutput(BaseModel):
    """Output schema for get_paper_content tool."""

    zotero_key: str
    title: str
    content: str
    truncated: bool = False


# ---------------------------------------------------------------------------
# Internal Search Functions
# ---------------------------------------------------------------------------


async def _semantic_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Semantic search using ES KNN vector search on L2 summaries."""
    store_manager = get_store_manager()

    try:
        query_embedding = await store_manager.embedding.embed(query)

        # Use ES KNN search on L2 summaries (papers have embeddings there)
        results = await store_manager.es_stores.store.knn_search(
            embedding=query_embedding,
            k=limit * 2,  # Get extra for deduplication
            compression_level=2,  # L2 has paper summaries with embeddings
        )

        search_results: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for record, score in results:
            zotero_key = record.zotero_key
            if not zotero_key or zotero_key in seen_keys:
                continue

            seen_keys.add(zotero_key)
            metadata = record.metadata or {}

            search_results.append({
                "zotero_key": zotero_key,
                "score": score,
                "title": metadata.get("title", "Unknown"),
                "year": metadata.get("year", 0),
                "authors": metadata.get("authors", []),
            })

        return search_results[:limit]

    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []


async def _keyword_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Keyword search using Elasticsearch text matching."""
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
            # Normalize score to 0-1 range
            score = min(1.0, getattr(r, "_score", 1.0) / 10.0)

            metadata = r.metadata or {}
            search_results.append({
                "zotero_key": r.zotero_key,
                "score": score,
                "title": metadata.get("title", "Unknown"),
                "year": metadata.get("year", 0),
                "authors": metadata.get("authors", []),
            })

        return search_results[:limit]

    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []


def _merge_search_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Merge results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}
    k = 60  # RRF constant

    # Add semantic scores
    for rank, result in enumerate(semantic_results):
        key = result["zotero_key"]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in metadata:
            metadata[key] = result

    # Add keyword scores
    for rank, result in enumerate(keyword_results):
        key = result["zotero_key"]
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in metadata:
            metadata[key] = result

    # Sort by combined score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    merged_results: list[dict[str, Any]] = []
    for key in sorted_keys[:limit]:
        result = metadata[key].copy()
        result["score"] = scores[key]
        merged_results.append(result)

    return merged_results


def _format_authors(authors: list[str]) -> str:
    """Format author list to 'Smith et al.' style."""
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0].split()[-1] if authors[0] else "Unknown"
    if len(authors) == 2:
        return f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    return f"{authors[0].split()[-1]} et al."


async def _enrich_with_zotero_metadata(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich search results with Zotero metadata for papers missing info."""
    store_manager = get_store_manager()

    # Find papers needing enrichment
    needs_enrichment = [
        r for r in results
        if r.get("title") in ("Unknown", None, "") or r.get("year") in (0, None)
    ]

    if not needs_enrichment:
        return results

    # Fetch Zotero metadata in parallel
    async def fetch_one(result: dict) -> tuple[str, dict | None]:
        zotero_key = result["zotero_key"]
        try:
            item = await store_manager.zotero.get(zotero_key)
            if item:
                fields = item.fields or {}
                creators = item.creators or []
                authors = [
                    c.get('name', '') or f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
                    for c in creators
                    if c.get("creatorType") == "author"
                ]
                date_str = fields.get("date", "")
                year = date_str[:4] if date_str and len(date_str) >= 4 else ""
                return zotero_key, {
                    "title": fields.get("title", "Unknown"),
                    "year": year,
                    "authors": authors,
                }
            return zotero_key, None
        except Exception as e:
            logger.debug(f"Failed to fetch Zotero metadata for {zotero_key}: {e}")
            return zotero_key, None

    # Limit concurrent requests
    semaphore = asyncio.Semaphore(10)

    async def fetch_with_semaphore(result: dict) -> tuple[str, dict | None]:
        async with semaphore:
            return await fetch_one(result)

    tasks = [fetch_with_semaphore(r) for r in needs_enrichment]
    fetched = await asyncio.gather(*tasks)

    # Build lookup and enrich results
    zotero_metadata = {key: meta for key, meta in fetched if meta is not None}

    for result in results:
        zotero_key = result["zotero_key"]
        if zotero_key in zotero_metadata:
            meta = zotero_metadata[zotero_key]
            if result.get("title") in ("Unknown", None, ""):
                result["title"] = meta.get("title", "Unknown")
            if result.get("year") in (0, None):
                year_str = meta.get("year", "")
                result["year"] = int(year_str) if year_str and year_str.isdigit() else 0
            if not result.get("authors"):
                result["authors"] = meta.get("authors", [])

    return results


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def search_papers(query: str, limit: int = 10) -> dict:
    """Search available papers by topic using hybrid search.

    Combines semantic (embedding) and keyword (Elasticsearch) search
    for best results on both conceptual queries and specific terms.

    Returns brief metadata for papers matching the query.
    Use get_paper_content to fetch detailed content for specific papers.

    Args:
        query: Topic, keyword, or concept to search for
        limit: Maximum papers to return (default 10, max 20)

    Returns:
        Dict with query, total_found, and list of papers with:
        - zotero_key: Citation key for [@KEY] format
        - title: Paper title
        - year: Publication year
        - authors: Brief author string (e.g., "Smith et al.")
        - relevance: Combined relevance score (0-1)
    """
    limit = clamp_limit(limit, min_val=1, max_val=20)

    # Run both search methods
    semantic_results = await _semantic_search(query, limit)
    keyword_results = await _keyword_search(query, limit)

    # Merge using RRF
    merged = _merge_search_results(semantic_results, keyword_results, limit)

    # Filter by minimum relevance
    merged = [r for r in merged if r["score"] >= MINIMUM_RELEVANCE_THRESHOLD]

    # Enrich with Zotero metadata
    merged = await _enrich_with_zotero_metadata(merged)

    # Build output
    papers = [
        PaperResult(
            zotero_key=r["zotero_key"],
            title=r.get("title", "Unknown")[:100],
            year=r.get("year", 0),
            authors=_format_authors(r.get("authors", [])),
            relevance=round(r["score"], 3),
        )
        for r in merged
    ]

    logger.info(f"search_papers('{query[:30]}...'): {len(papers)} results")

    return output_dict(
        PaperSearchOutput(
            query=query,
            total_found=len(papers),
            papers=papers,
        )
    )


@tool
async def get_paper_content(zotero_key: str, max_chars: int = 10000) -> dict:
    """Fetch detailed content for a specific paper.

    Returns the 10:1 compressed summary (L2) which captures key content
    while fitting in context. Use after search_papers identifies relevant papers.

    Args:
        zotero_key: Paper citation key from search_papers results (8 alphanumeric chars)
        max_chars: Maximum content length (default 10000, max 20000)

    Returns:
        Dict with:
        - zotero_key: Paper citation key
        - title: Full paper title
        - content: L2 10:1 compressed content
        - truncated: Whether content was cut to fit max_chars
    """
    max_chars = clamp_limit(max_chars, min_val=1000, max_val=20000)
    store_manager = get_store_manager()

    try:
        # Try L2 first (10:1 compressed summaries)
        es_query = {
            "bool": {
                "must": [
                    {"term": {"zotero_key.keyword": zotero_key}},
                ],
                "filter": [
                    {"term": {"compression_level": 2}},
                ],
            }
        }

        records = await store_manager.es_stores.store.search(query=es_query, size=1)

        if records:
            record = records[0]
            content = record.content or ""
            metadata = record.metadata or {}
            title = metadata.get("title", "Unknown")

            truncated = len(content) > max_chars
            if truncated:
                content = content[:max_chars] + "\n\n[... content truncated ...]"

            logger.info(
                f"get_paper_content({zotero_key}): {len(content)} chars, truncated={truncated}"
            )

            return output_dict(
                PaperContentOutput(
                    zotero_key=zotero_key,
                    title=title,
                    content=content,
                    truncated=truncated,
                )
            )

        # Fall back to L1 if L2 not available
        es_query["bool"]["filter"] = [{"term": {"compression_level": 1}}]
        records = await store_manager.es_stores.store.search(query=es_query, size=1)

        if records:
            record = records[0]
            content = record.content or ""
            metadata = record.metadata or {}
            title = metadata.get("title", "Unknown")

            truncated = len(content) > max_chars
            if truncated:
                content = content[:max_chars] + "\n\n[... content truncated ...]"

            return output_dict(
                PaperContentOutput(
                    zotero_key=zotero_key,
                    title=title,
                    content=content,
                    truncated=truncated,
                )
            )

        # Try to get metadata from Zotero as fallback
        item = await store_manager.zotero.get(zotero_key)
        if item:
            title = item.fields.get("title", "Unknown") if item.fields else "Unknown"
            return output_dict(
                PaperContentOutput(
                    zotero_key=zotero_key,
                    title=title,
                    content="No detailed content available in store. Paper exists in Zotero library.",
                    truncated=False,
                )
            )

        # Not found anywhere
        return output_dict(
            PaperContentOutput(
                zotero_key=zotero_key,
                title="Unknown",
                content=f"Paper with key {zotero_key} not found in store or Zotero.",
                truncated=False,
            )
        )

    except Exception as e:
        logger.error(f"get_paper_content failed for {zotero_key}: {e}")
        return output_dict(
            PaperContentOutput(
                zotero_key=zotero_key,
                title="Error",
                content=f"Failed to fetch paper content: {e}",
                truncated=False,
            )
        )
