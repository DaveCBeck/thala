"""
search_memory - Cross-store semantic search tool for LangChain.

Searches across: top_of_mind, coherence, who_i_was, store
"""

import logging
from typing import Any, Literal, Optional
from uuid import UUID

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import StoreManager, get_store_manager

logger = logging.getLogger(__name__)


class MemorySearchResult(BaseModel):
    """Individual search result from memory."""

    id: str
    source_store: str
    content: str
    score: Optional[float] = None
    zotero_key: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class SearchMemoryInput(BaseModel):
    """Input schema for search_memory tool."""

    query: str = Field(description="What to search for in memory")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max results per store (default 10)",
    )
    stores: Optional[list[Literal["top_of_mind", "coherence", "who_i_was", "store"]]] = Field(
        default=None,
        description="Specific stores to search. Defaults to all except who_i_was.",
    )
    include_historical: bool = Field(
        default=False,
        description="Include who_i_was (historical versions). Defaults to False.",
    )


class SearchMemoryOutput(BaseModel):
    """Output schema for search_memory tool."""

    query: str
    total_results: int
    results: list[MemorySearchResult]


class SearchMemoryTool(BaseTool):
    """
    Search across the memory system for relevant information.

    Use this when you need to:
    - Recall something you might have learned before
    - Find beliefs, preferences, or identity information
    - Look up stored knowledge on a topic

    Returns ranked results from multiple memory stores.
    """

    name: str = "search_memory"
    description: str = """Search across the memory system for relevant information.

Use this when you need to:
- Recall something you might have learned before
- Find beliefs, preferences, or identity information (coherence store)
- Look up stored knowledge on a topic

Returns results from multiple memory stores with source attribution."""

    args_schema: type[BaseModel] = SearchMemoryInput

    # Store manager - can be injected or uses default
    store_manager: StoreManager = Field(default_factory=get_store_manager)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        limit: int = 10,
        stores: Optional[list[str]] = None,
        include_historical: bool = False,
    ) -> dict[str, Any]:
        """Sync wrapper - raises error, use async version."""
        raise NotImplementedError("SearchMemoryTool only supports async. Use ainvoke().")

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        stores: Optional[list[str]] = None,
        include_historical: bool = False,
    ) -> dict[str, Any]:
        """
        Search across memory stores.

        Args:
            query: Search query
            limit: Max results per store
            stores: Which stores to search (default: all except who_i_was)
            include_historical: Include who_i_was store

        Returns:
            Dict with query, total_results, and results list
        """
        results: list[MemorySearchResult] = []

        # Default stores if not specified
        if stores is None:
            stores = ["top_of_mind", "coherence", "store"]
            if include_historical:
                stores.append("who_i_was")

        # 1. Semantic search in top_of_mind (vector similarity)
        if "top_of_mind" in stores:
            try:
                query_embedding = await self.store_manager.embedding.embed(query)
                chroma_results = await self.store_manager.chroma.search(
                    query_embedding=query_embedding,
                    n_results=limit,
                )
                for r in chroma_results:
                    results.append(MemorySearchResult(
                        id=str(r["id"]),
                        source_store="top_of_mind",
                        content=r["document"] or "",
                        score=1 - r["distance"],  # Convert distance to similarity
                        zotero_key=r["metadata"].get("zotero_key") if r["metadata"] else None,
                        metadata=r["metadata"] or {},
                    ))
                logger.debug(f"top_of_mind returned {len(chroma_results)} results")
            except Exception as e:
                logger.warning(f"top_of_mind search failed: {e}")

        # 2. Text search in coherence (beliefs/preferences)
        if "coherence" in stores:
            try:
                coherence_results = await self.store_manager.es_stores.coherence.search(
                    query={"match": {"content": query}},
                    size=limit,
                )
                for r in coherence_results:
                    results.append(MemorySearchResult(
                        id=str(r.id),
                        source_store="coherence",
                        content=r.content,
                        score=None,  # ES scores not directly comparable to vector distances
                        zotero_key=r.zotero_key,
                        metadata={
                            "category": r.category,
                            "confidence": r.confidence,
                        },
                    ))
                logger.debug(f"coherence returned {len(coherence_results)} results")
            except Exception as e:
                logger.warning(f"coherence search failed: {e}")

        # 3. Text search in store (knowledge base)
        if "store" in stores:
            try:
                store_results = await self.store_manager.es_stores.store.search(
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
                history_results = await self.store_manager.es_stores.who_i_was.search(
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

        # TODO: Priority/ranking strategy across stores
        # Current behavior: Results are grouped by store in search order
        # Options to consider:
        # - Interleave by normalized score (requires score normalization)
        # - top_of_mind (semantic) first, then text matches
        # - Deduplicate by content hash
        # - Boost coherence records (beliefs are important)

        output = SearchMemoryOutput(
            query=query,
            total_results=len(results),
            results=results,
        )

        return output.model_dump(mode="json")
