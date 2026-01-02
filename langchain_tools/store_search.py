"""
Store-specific search tools for LangChain.

Individual search tools for each store with appropriate filters:
- search_store: Main knowledge store (language, compression, type)
- search_coherence: Beliefs/preferences (category, confidence)
- search_top_of_mind: Active projects (language, type)
- search_history: Historical versions (original_store)
- search_forgotten: Archived content (forgotten_reason)
"""

import logging
from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from .base import get_store_manager

logger = logging.getLogger(__name__)


class StoreSearchResult(BaseModel):
    """Individual search result."""

    id: str
    content: str
    score: Optional[float] = None
    language_code: Optional[str] = None
    zotero_key: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class StoreSearchOutput(BaseModel):
    """Output schema for store search tools."""

    query: str
    store: str
    total_results: int
    results: list[StoreSearchResult]


def _build_es_query(
    query: str,
    language_code: Optional[str] = None,
    record_type: Optional[str] = None,
    extra_filters: Optional[list[dict]] = None,
) -> dict:
    """Build an Elasticsearch bool query with filters."""
    must = [{"match": {"content": query}}]
    filters = []

    if language_code:
        filters.append({"term": {"language_code": language_code}})

    if record_type:
        filters.append({"term": {"metadata.type": record_type}})

    if extra_filters:
        filters.extend(extra_filters)

    if filters:
        return {"bool": {"must": must, "filter": filters}}
    return {"match": {"content": query}}


@tool
async def search_store(
    query: str,
    limit: int = 10,
    language_code: Optional[str] = None,
    record_type: Optional[str] = None,
    compression_level: Optional[int] = None,
    source_type: Optional[Literal["external", "internal"]] = None,
) -> dict:
    """Search the main knowledge store for relevant information.

    The main store contains all knowledge: research reports, documents,
    compressions, and multi-language research results.

    Args:
        query: What to search for
        limit: Max results (default 10, max 50)
        language_code: Filter by ISO 639-1 code (e.g., 'en', 'es', 'de')
        record_type: Filter by type (e.g., 'research_report', 'multi_lang_synthesis')
        compression_level: Filter by compression depth (0=original, 1+=compressed)
        source_type: Filter by origin ('external' from Zotero, 'internal' system-generated)
    """
    store_manager = get_store_manager()
    limit = min(max(1, limit), 50)

    # Build filters
    filters = []
    if compression_level is not None:
        filters.append({"term": {"compression_level": compression_level}})
    if source_type:
        filters.append({"term": {"source_type": source_type}})

    es_query = _build_es_query(
        query,
        language_code=language_code,
        record_type=record_type,
        extra_filters=filters if filters else None,
    )

    results = []
    try:
        records = await store_manager.es_stores.store.search(query=es_query, size=limit)
        for r in records:
            results.append(StoreSearchResult(
                id=str(r.id),
                content=r.content,
                language_code=r.language_code,
                zotero_key=r.zotero_key,
                metadata={
                    "type": r.metadata.get("type"),
                    "compression_level": r.compression_level,
                    "source_type": r.source_type.value if hasattr(r.source_type, "value") else str(r.source_type),
                },
            ))
    except Exception as e:
        logger.warning(f"store search failed: {e}")

    return StoreSearchOutput(
        query=query,
        store="store",
        total_results=len(results),
        results=results,
    ).model_dump(mode="json")


@tool
async def search_coherence(
    query: str,
    limit: int = 10,
    category: Optional[str] = None,
    min_confidence: float = 0.3,
) -> dict:
    """Search beliefs, preferences, and identity information.

    The coherence store contains the system's understanding of who you are,
    what you believe, and your preferences - each with a confidence score.

    Args:
        query: What to search for
        limit: Max results (default 10, max 50)
        category: Filter by category ('belief', 'preference', 'identity', 'goal')
        min_confidence: Minimum confidence threshold (0-1, default 0.3)
    """
    store_manager = get_store_manager()
    limit = min(max(1, limit), 50)

    # Build query
    filters = []
    if category:
        filters.append({"term": {"category": category}})
    if min_confidence > 0:
        filters.append({"range": {"confidence": {"gte": min_confidence}}})

    es_query = _build_es_query(query, extra_filters=filters if filters else None)

    results = []
    try:
        records = await store_manager.es_stores.coherence.search(query=es_query, size=limit)
        for r in records:
            confidence = r.confidence if hasattr(r, "confidence") else None
            results.append(StoreSearchResult(
                id=str(r.id),
                content=r.content,
                score=confidence,
                language_code=getattr(r, "language_code", None),
                metadata={
                    "category": r.category if hasattr(r, "category") else None,
                    "confidence": confidence,
                },
            ))
    except Exception as e:
        logger.warning(f"coherence search failed: {e}")

    return StoreSearchOutput(
        query=query,
        store="coherence",
        total_results=len(results),
        results=results,
    ).model_dump(mode="json")


@tool
async def search_top_of_mind(
    query: str,
    limit: int = 10,
    language_code: Optional[str] = None,
    record_type: Optional[str] = None,
    min_similarity: float = 0.5,
) -> dict:
    """Search active projects and current context using semantic similarity.

    The top_of_mind store contains what's currently relevant - active projects,
    recent research, and current context. Uses vector similarity search.

    Args:
        query: What to search for (semantic similarity)
        limit: Max results (default 10, max 50)
        language_code: Filter by ISO 639-1 code (e.g., 'en', 'es')
        record_type: Filter by type (e.g., 'wrapped_research', 'multi_lang_synthesis')
        min_similarity: Minimum similarity threshold (0-1, default 0.5)
    """
    store_manager = get_store_manager()
    limit = min(max(1, limit), 50)

    # Build Chroma where filter
    where = {}
    if language_code:
        where["language_code"] = language_code
    if record_type:
        where["type"] = record_type

    results = []
    try:
        query_embedding = await store_manager.embedding.embed(query)
        chroma_results = await store_manager.chroma.search(
            query_embedding=query_embedding,
            n_results=limit,
            where=where if where else None,
        )

        for r in chroma_results:
            similarity = 1 - r["distance"]
            if similarity < min_similarity:
                continue

            metadata = r.get("metadata") or {}
            results.append(StoreSearchResult(
                id=str(r["id"]),
                content=r["document"] or "",
                score=similarity,
                language_code=metadata.get("language_code"),
                zotero_key=metadata.get("zotero_key"),
                metadata=metadata,
            ))
    except Exception as e:
        logger.warning(f"top_of_mind search failed: {e}")

    return StoreSearchOutput(
        query=query,
        store="top_of_mind",
        total_results=len(results),
        results=results,
    ).model_dump(mode="json")


@tool
async def search_history(
    query: str,
    limit: int = 10,
    original_store: Optional[Literal["coherence", "top_of_mind"]] = None,
) -> dict:
    """Search historical versions of changed records.

    The who_i_was store contains previous versions of beliefs and context
    that have been superseded. Useful for understanding how things evolved.

    Args:
        query: What to search for in previous content
        limit: Max results (default 10, max 50)
        original_store: Filter by source ('coherence' or 'top_of_mind')
    """
    store_manager = get_store_manager()
    limit = min(max(1, limit), 50)

    # Search in previous_data.content
    must = [{"match": {"previous_data.content": query}}]
    filters = []
    if original_store:
        filters.append({"term": {"original_store": original_store}})

    es_query = {"bool": {"must": must, "filter": filters}} if filters else {"match": {"previous_data.content": query}}

    results = []
    try:
        records = await store_manager.es_stores.who_i_was.search(query=es_query, size=limit)
        for r in records:
            prev_data = r.previous_data or {}
            results.append(StoreSearchResult(
                id=str(r.id),
                content=prev_data.get("content", ""),
                language_code=prev_data.get("language_code"),
                zotero_key=prev_data.get("zotero_key"),
                metadata={
                    "supersedes": str(r.supersedes),
                    "reason": r.reason,
                    "original_store": r.original_store,
                },
            ))
    except Exception as e:
        logger.warning(f"who_i_was search failed: {e}")

    return StoreSearchOutput(
        query=query,
        store="who_i_was",
        total_results=len(results),
        results=results,
    ).model_dump(mode="json")


@tool
async def search_forgotten(
    query: str,
    limit: int = 10,
    original_store: Optional[str] = None,
    forgotten_reason: Optional[str] = None,
) -> dict:
    """Search archived/forgotten content.

    The forgotten store contains records that were deliberately archived
    with a reason. Useful for finding things that were once relevant.

    Args:
        query: What to search for
        limit: Max results (default 10, max 50)
        original_store: Filter by original source ('coherence', 'store')
        forgotten_reason: Filter by reason for archiving (partial match)
    """
    store_manager = get_store_manager()
    limit = min(max(1, limit), 50)

    must = [{"match": {"content": query}}]
    filters = []
    if original_store:
        filters.append({"term": {"original_store": original_store}})
    if forgotten_reason:
        filters.append({"match": {"forgotten_reason": forgotten_reason}})

    es_query = {"bool": {"must": must, "filter": filters}} if filters else {"match": {"content": query}}

    results = []
    try:
        records = await store_manager.es_stores.forgotten.search(query=es_query, size=limit)
        for r in records:
            results.append(StoreSearchResult(
                id=str(r.id),
                content=r.content,
                language_code=getattr(r, "language_code", None),
                zotero_key=r.zotero_key,
                metadata={
                    "forgotten_reason": r.forgotten_reason,
                    "forgotten_at": r.forgotten_at.isoformat() if r.forgotten_at else None,
                    "original_store": r.original_store,
                },
            ))
    except Exception as e:
        logger.warning(f"forgotten search failed: {e}")

    return StoreSearchOutput(
        query=query,
        store="forgotten",
        total_results=len(results),
        results=results,
    ).model_dump(mode="json")
