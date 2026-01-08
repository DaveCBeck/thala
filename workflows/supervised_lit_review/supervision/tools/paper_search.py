"""Paper search tools for LLM use in supervision loops.

Provides two-stage retrieval:
1. search_papers: Hybrid (semantic + keyword) search returning compact metadata
2. get_paper_content: Fetch detailed L2 content for specific papers
"""

import logging
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_tools.base import get_store_manager
from workflows.academic_lit_review.state import PaperSummary
from workflows.supervised_lit_review.supervision.store_query import (
    SupervisionStoreQuery,
)

logger = logging.getLogger(__name__)

# Minimum relevance score for search results to be returned
# Papers below this threshold are filtered out to prevent citation drift
MINIMUM_RELEVANCE_THRESHOLD = 0.5


class PaperSearchResult(BaseModel):
    """Compact paper metadata for search results."""

    doi: str = Field(description="Paper DOI")
    title: str = Field(description="Paper title")
    year: int = Field(description="Publication year")
    authors: str = Field(description="Authors in 'Smith et al.' format")
    relevance: float = Field(description="Relevance score 0-1")
    zotero_key: str = Field(description="Citation key for [@KEY] format")


class SearchPapersOutput(BaseModel):
    """Output from search_papers tool."""

    query: str = Field(description="The search query")
    total_found: int = Field(description="Number of papers found")
    papers: list[PaperSearchResult] = Field(description="Matching papers")


class PaperContentOutput(BaseModel):
    """Output from get_paper_content tool."""

    doi: str = Field(description="Paper DOI")
    title: str = Field(description="Paper title")
    content: str = Field(description="L2 10:1 compressed content")
    key_findings: list[str] = Field(description="Key findings from extraction")
    truncated: bool = Field(description="Whether content was truncated")


def _format_authors(authors: list[str]) -> str:
    """Format author list to 'Smith et al.' style."""
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0].split()[-1] if authors[0] else "Unknown"
    if len(authors) == 2:
        return f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    return f"{authors[0].split()[-1]} et al."


async def _semantic_search(
    query: str,
    paper_summaries: dict[str, PaperSummary],
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Semantic search using embeddings.

    Returns list of (doi, similarity_score) tuples.
    """
    store_manager = get_store_manager()

    # Get DOIs that have ES records (can be searched in Chroma)
    searchable_dois = [
        doi for doi, s in paper_summaries.items() if s.get("es_record_id")
    ]

    if not searchable_dois:
        return []

    try:
        query_embedding = await store_manager.embedding.embed(query)

        # Search Chroma for semantically similar content
        results = await store_manager.chroma.search(
            query_embedding=query_embedding,
            n_results=limit * 2,  # Get extra for filtering
            where=None,  # No filter - search all
        )

        # Map results back to DOIs via zotero_key
        doi_scores: list[tuple[str, float]] = []
        key_to_doi = {
            s.get("zotero_key"): doi
            for doi, s in paper_summaries.items()
            if s.get("zotero_key")
        }

        for r in results:
            metadata = r.get("metadata", {})
            zotero_key = metadata.get("zotero_key")
            if zotero_key and zotero_key in key_to_doi:
                doi = key_to_doi[zotero_key]
                similarity = 1 - r.get("distance", 1)  # Convert distance to similarity
                doi_scores.append((doi, similarity))

        return doi_scores[:limit]

    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []


async def _keyword_search(
    query: str,
    paper_summaries: dict[str, PaperSummary],
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Keyword search using Elasticsearch text matching.

    Returns list of (doi, relevance_score) tuples.
    """
    store_manager = get_store_manager()

    try:
        es_query = {"match": {"content": query}}
        records = await store_manager.es_stores.store.search(
            query=es_query,
            size=limit * 2,
        )

        # Map results back to DOIs
        zotero_key_to_doi = {
            s.get("zotero_key"): doi
            for doi, s in paper_summaries.items()
            if s.get("zotero_key")
        }

        doi_scores: list[tuple[str, float]] = []
        for r in records:
            if r.zotero_key and r.zotero_key in zotero_key_to_doi:
                doi = zotero_key_to_doi[r.zotero_key]
                # Normalize score to 0-1 range (ES scores vary widely)
                score = min(1.0, getattr(r, "_score", 1.0) / 10.0)
                doi_scores.append((doi, score))

        return doi_scores[:limit]

    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []


def _merge_search_results(
    semantic_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Merge and rank results from both search methods.

    Uses reciprocal rank fusion for fair combination.
    """
    scores: dict[str, float] = {}
    k = 60  # RRF constant

    # Add semantic scores using RRF
    for rank, (doi, _) in enumerate(semantic_results):
        scores[doi] = scores.get(doi, 0) + 1.0 / (k + rank + 1)

    # Add keyword scores using RRF
    for rank, (doi, _) in enumerate(keyword_results):
        scores[doi] = scores.get(doi, 0) + 1.0 / (k + rank + 1)

    # Sort by combined score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:limit]


def create_paper_tools(
    paper_summaries: dict[str, PaperSummary],
    store_query: SupervisionStoreQuery,
) -> list:
    """Create paper search tools scoped to the current paper set.

    Tools are closures over paper_summaries and store_query,
    ensuring they only search/fetch papers available in the current review.

    Args:
        paper_summaries: DOI -> PaperSummary mapping from workflow state
        store_query: SupervisionStoreQuery instance for content fetching

    Returns:
        List of LangChain tools [search_papers, get_paper_content]
    """

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
            - doi: Paper DOI for use with get_paper_content
            - title: Paper title
            - year: Publication year
            - authors: Brief author string
            - relevance: Combined relevance score (0-1)
            - zotero_key: Citation key for [@KEY] format
        """
        limit = min(limit, 20)  # Cap at 20

        # Run both search methods
        semantic_results = await _semantic_search(query, paper_summaries, limit)
        keyword_results = await _keyword_search(query, paper_summaries, limit)

        # Merge results using reciprocal rank fusion
        merged = _merge_search_results(semantic_results, keyword_results, limit)

        # Filter by minimum relevance threshold to prevent citation drift
        merged = [
            (doi, score) for doi, score in merged
            if score >= MINIMUM_RELEVANCE_THRESHOLD
        ]

        # Build output with paper metadata
        papers: list[dict[str, Any]] = []
        for doi, score in merged:
            summary = paper_summaries.get(doi)
            if not summary:
                continue

            papers.append({
                "doi": doi,
                "title": summary.get("title", "Unknown")[:100],
                "year": summary.get("year", 0),
                "authors": _format_authors(summary.get("authors", [])),
                "relevance": round(score, 3),
                "zotero_key": summary.get("zotero_key", ""),
            })

        logger.info(f"search_papers('{query[:30]}...'): {len(papers)} results")

        return {
            "query": query,
            "total_found": len(papers),
            "papers": papers,
        }

    @tool
    async def get_paper_content(doi: str, max_chars: int = 10000) -> dict:
        """Fetch detailed content for a specific paper.

        Returns the 10:1 compressed summary (L2) which captures key content
        while fitting in context. Use after search_papers identifies relevant papers.

        Args:
            doi: Paper DOI from search_papers results
            max_chars: Maximum content length (default 10000, max 20000)

        Returns:
            Dict with:
            - doi: Paper DOI
            - title: Full paper title
            - content: L2 10:1 compressed content
            - key_findings: List of key findings from extraction
            - truncated: Whether content was cut to fit max_chars
        """
        max_chars = min(max_chars, 20000)  # Cap at 20K

        summary = paper_summaries.get(doi)
        if not summary:
            return {
                "doi": doi,
                "title": "Unknown",
                "content": f"Paper with DOI {doi} not found in available papers.",
                "key_findings": [],
                "truncated": False,
            }

        # Fetch L2 content
        content = await store_query.get_paper_content(doi, compression_level=2)

        if not content:
            # Fall back to metadata
            content = (
                f"No detailed content available.\n\n"
                f"Short summary: {summary.get('short_summary', 'N/A')}\n"
                f"Methodology: {summary.get('methodology', 'N/A')}"
            )

        # Truncate if needed
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars] + "\n\n[... content truncated ...]"

        logger.info(
            f"get_paper_content({doi[:20]}...): {len(content)} chars, truncated={truncated}"
        )

        return {
            "doi": doi,
            "title": summary.get("title", "Unknown"),
            "content": content,
            "key_findings": summary.get("key_findings", [])[:5],
            "truncated": truncated,
        }

    return [search_papers, get_paper_content]
