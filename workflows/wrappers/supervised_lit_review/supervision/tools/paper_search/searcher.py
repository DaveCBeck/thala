"""Main paper search logic and tool creation."""

import logging
from typing import Any

from langchain_core.tools import tool

from langchain_tools.base import get_store_manager
from core.stores import ZoteroStore
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


def create_paper_tools(store_query: "SupervisionStoreQuery") -> list:
    """Create paper search tools for section editing.

    Tools search the full ES/Chroma corpus without filtering.

    Args:
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
            - zotero_key: Citation key for [@KEY] format
            - title: Paper title
            - year: Publication year
            - authors: Brief author string
            - relevance: Combined relevance score (0-1)
        """
        limit = min(limit, 20)  # Cap at 20

        # Run both search methods (no filtering through paper_summaries)
        semantic_results = await semantic_search(query, limit)
        keyword_results = await keyword_search(query, limit)

        # Merge results using reciprocal rank fusion
        merged = merge_search_results(semantic_results, keyword_results, limit)

        # Filter by minimum relevance threshold to prevent citation drift
        merged = [r for r in merged if r["score"] >= MINIMUM_RELEVANCE_THRESHOLD]

        # Build output with paper metadata
        papers: list[dict[str, Any]] = []
        for result in merged:
            papers.append({
                "zotero_key": result["zotero_key"],
                "title": result.get("title", "Unknown")[:100],
                "year": result.get("year", 0),
                "authors": format_authors(result.get("authors", [])),
                "relevance": round(result["score"], 3),
            })

        logger.info(f"search_papers('{query[:30]}...'): {len(papers)} results")

        return {
            "query": query,
            "total_found": len(papers),
            "papers": papers,
        }

    @tool
    async def get_paper_content(zotero_key: str, max_chars: int = 10000) -> dict:
        """Fetch detailed content for a specific paper.

        Returns the 10:1 compressed summary (L2) which captures key content
        while fitting in context. Use after search_papers identifies relevant papers.

        Args:
            zotero_key: Paper citation key from search_papers results
            max_chars: Maximum content length (default 10000, max 20000)

        Returns:
            Dict with:
            - zotero_key: Paper citation key
            - title: Full paper title
            - content: L2 10:1 compressed content
            - truncated: Whether content was cut to fit max_chars
        """
        max_chars = min(max_chars, 20000)  # Cap at 20K

        # Fetch content from ES
        content, metadata = await store_query.get_paper_content(zotero_key)

        if not content:
            # Try to get metadata from Zotero as fallback
            zotero_metadata = await store_query.get_zotero_metadata(zotero_key)
            if zotero_metadata:
                return {
                    "zotero_key": zotero_key,
                    "title": zotero_metadata.get("title", "Unknown"),
                    "content": f"No detailed content available in store. Paper exists in Zotero library.",
                    "truncated": False,
                }
            return {
                "zotero_key": zotero_key,
                "title": "Unknown",
                "content": f"Paper with key {zotero_key} not found in store or Zotero.",
                "truncated": False,
            }

        # Truncate if needed
        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars] + "\n\n[... content truncated ...]"

        title = metadata.get("title", "Unknown") if metadata else "Unknown"

        logger.info(
            f"get_paper_content({zotero_key}): {len(content)} chars, truncated={truncated}"
        )

        return {
            "zotero_key": zotero_key,
            "title": title,
            "content": content,
            "truncated": truncated,
        }

    return [search_papers, get_paper_content]
