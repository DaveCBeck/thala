"""
search_memory - Cross-store semantic search tool for LangChain.

Searches across: top_of_mind, coherence, who_i_was, store
"""

import logging
from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from .base import get_store_manager

logger = logging.getLogger(__name__)

# Minimum similarity score for vector search results (0.0-1.0)
# Results below this threshold are considered irrelevant and filtered out
# 0.5 is a reasonable default - adjust based on testing
MIN_VECTOR_SIMILARITY = 0.5

# Minimum confidence for coherence records (beliefs/preferences)
# Records below this threshold are filtered out
MIN_COHERENCE_CONFIDENCE = 0.3


class MemorySearchResult(BaseModel):
    """Individual search result from memory."""

    id: str
    source_store: str
    content: str
    score: Optional[float] = None
    zotero_key: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SearchMemoryOutput(BaseModel):
    """Output schema for search_memory tool."""

    query: str
    total_results: int
    results: list[MemorySearchResult]


@tool
async def search_memory(
    query: str,
    limit: int = 10,
    stores: Optional[list[Literal["top_of_mind", "coherence", "who_i_was", "store"]]] = None,
    include_historical: bool = False,
) -> dict:
    """Search across the memory system for relevant information.

    Use this when you need to:
    - Recall something you might have learned before
    - Find beliefs, preferences, or identity information (coherence store)
    - Look up stored knowledge on a topic

    Returns results from multiple memory stores with source attribution.

    Args:
        query: What to search for in memory
        limit: Max results per store (default 10, max 50)
        stores: Specific stores to search. Defaults to all except who_i_was.
        include_historical: Include who_i_was (historical versions). Defaults to False.
    """
    store_manager = get_store_manager()
    results: list[MemorySearchResult] = []

    # Clamp limit
    limit = min(max(1, limit), 50)

    # Default stores if not specified
    if stores is None:
        stores = ["top_of_mind", "coherence", "store"]
        if include_historical:
            stores.append("who_i_was")

    # 1. Semantic search in top_of_mind (vector similarity)
    if "top_of_mind" in stores:
        try:
            query_embedding = await store_manager.embedding.embed(query)
            chroma_results = await store_manager.chroma.search(
                query_embedding=query_embedding,
                n_results=limit,
            )
            filtered_count = 0
            for r in chroma_results:
                similarity = 1 - r["distance"]  # Convert distance to similarity

                # Filter out low-relevance results
                if similarity < MIN_VECTOR_SIMILARITY:
                    filtered_count += 1
                    logger.debug(
                        f"Filtered top_of_mind result with low similarity: {similarity:.3f} < {MIN_VECTOR_SIMILARITY}"
                    )
                    continue

                results.append(MemorySearchResult(
                    id=str(r["id"]),
                    source_store="top_of_mind",
                    content=r["document"] or "",
                    score=similarity,
                    zotero_key=r["metadata"].get("zotero_key") if r["metadata"] else None,
                    metadata=r["metadata"] or {},
                ))

            accepted = len(chroma_results) - filtered_count
            logger.debug(
                f"top_of_mind: {accepted}/{len(chroma_results)} results passed similarity filter (>= {MIN_VECTOR_SIMILARITY})"
            )
        except Exception as e:
            logger.warning(f"top_of_mind search failed: {e}")

    # 2. Text search in coherence (beliefs/preferences)
    if "coherence" in stores:
        try:
            coherence_results = await store_manager.es_stores.coherence.search(
                query={"match": {"content": query}},
                size=limit,
            )
            filtered_count = 0
            for r in coherence_results:
                # Filter by confidence if available
                confidence = r.confidence if hasattr(r, 'confidence') else None
                if confidence is not None and confidence < MIN_COHERENCE_CONFIDENCE:
                    filtered_count += 1
                    logger.debug(
                        f"Filtered coherence result with low confidence: {confidence:.2f} < {MIN_COHERENCE_CONFIDENCE}"
                    )
                    continue

                results.append(MemorySearchResult(
                    id=str(r.id),
                    source_store="coherence",
                    content=r.content,
                    score=confidence,  # Use confidence as score for coherence records
                    zotero_key=r.zotero_key,
                    metadata={
                        "category": r.category,
                        "confidence": confidence,
                    },
                ))

            accepted = len(coherence_results) - filtered_count
            logger.debug(
                f"coherence: {accepted}/{len(coherence_results)} results passed confidence filter (>= {MIN_COHERENCE_CONFIDENCE})"
            )
        except Exception as e:
            logger.warning(f"coherence search failed: {e}")

    # 3. Text search in store (knowledge base)
    if "store" in stores:
        try:
            store_results = await store_manager.es_stores.store.search(
                query={"match": {"content": query}},
                size=limit,
            )
            for r in store_results:
                results.append(MemorySearchResult(
                    id=str(r.id),
                    source_store="store",
                    content=r.content,
                    score=None,
                    zotero_key=r.zotero_key,
                    metadata={
                        "compression_level": r.compression_level,
                        "source_type": r.source_type.value if hasattr(r.source_type, 'value') else str(r.source_type),
                    },
                ))
            logger.debug(f"store returned {len(store_results)} results")
        except Exception as e:
            logger.warning(f"store search failed: {e}")

    # 4. Historical search (optional)
    if include_historical and "who_i_was" in stores:
        try:
            history_results = await store_manager.es_stores.who_i_was.search(
                query={"match": {"previous_data.content": query}},
                size=limit,
            )
            for r in history_results:
                results.append(MemorySearchResult(
                    id=str(r.id),
                    source_store="who_i_was",
                    content=r.previous_data.get("content", "") if r.previous_data else "",
                    score=None,
                    zotero_key=r.previous_data.get("zotero_key") if r.previous_data else None,
                    metadata={
                        "supersedes": str(r.supersedes),
                        "reason": r.reason,
                        "original_store": r.original_store,
                    },
                ))
            logger.debug(f"who_i_was returned {len(history_results)} results")
        except Exception as e:
            logger.warning(f"who_i_was search failed: {e}")

    # Results are filtered by relevance thresholds (MIN_VECTOR_SIMILARITY, MIN_COHERENCE_CONFIDENCE)
    # and grouped by store in search order. Future improvements could include:
    # - Interleave by normalized score across all stores
    # - Deduplicate by content hash

    output = SearchMemoryOutput(
        query=query,
        total_results=len(results),
        results=results,
    )

    return output.model_dump(mode="json")
