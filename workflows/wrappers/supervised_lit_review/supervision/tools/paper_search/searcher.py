"""Main paper search logic and tool creation."""

import logging
from typing import Any

from langchain_core.tools import tool

from workflows.research.academic_lit_review.state import PaperSummary
from workflows.wrappers.supervised_lit_review.supervision.store_query import (
    SupervisionStoreQuery,
)
from .sources import semantic_search, keyword_search, merge_search_results

logger = logging.getLogger(__name__)

# Minimum relevance score for search results to be returned
# Papers below this threshold are filtered out to prevent citation drift
MINIMUM_RELEVANCE_THRESHOLD = 0.5


def format_authors(authors: list[str]) -> str:
    """Format author list to 'Smith et al.' style."""
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0].split()[-1] if authors[0] else "Unknown"
    if len(authors) == 2:
        return f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    return f"{authors[0].split()[-1]} et al."


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
        semantic_results = await semantic_search(query, paper_summaries, limit)
        keyword_results = await keyword_search(query, paper_summaries, limit)

        # Merge results using reciprocal rank fusion
        merged = merge_search_results(semantic_results, keyword_results, limit)

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
                "authors": format_authors(summary.get("authors", [])),
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
