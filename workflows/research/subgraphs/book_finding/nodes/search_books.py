"""
Book search node using book_search tool.

Searches for books matching the LLM recommendations using the
Library Genesis / Open Library API.
"""

import asyncio
import logging
from typing import Any

from langchain_tools.book_search import book_search
from workflows.research.subgraphs.book_finding.state import BookResult

logger = logging.getLogger(__name__)

# Search configuration - higher limit to improve PDF availability
MAX_RESULTS_PER_SEARCH = 15


async def _search_single_book(
    recommendation_title: str,
    author: str | None,
    language: str | None = None,
) -> BookResult | None:
    """Search for a single recommended book.

    Args:
        recommendation_title: Title of the recommended book
        author: Optional author name to improve search
        language: Optional language filter (e.g., "en", "es")

    Returns:
        BookResult if found, None otherwise
    """
    # Build search query
    query = recommendation_title
    if author:
        query = f"{recommendation_title} {author}"

    try:
        search_params = {
            "query": query,
            "limit": MAX_RESULTS_PER_SEARCH,
        }
        if language:
            search_params["language"] = language

        result = await book_search.ainvoke(search_params)

        books = result.get("results", [])
        if not books:
            logger.info(f"No results found for: {query}")
            return None

        # Prioritize PDF format, then take first result
        pdf_books = [b for b in books if b.get("format", "").lower() == "pdf"]
        best_book = pdf_books[0] if pdf_books else books[0]

        return BookResult(
            title=best_book.get("title", ""),
            authors=best_book.get("authors", "Unknown"),
            md5=best_book.get("md5", ""),
            url=best_book.get("url", ""),
            format=best_book.get("format", ""),
            size=best_book.get("size", ""),
            abstract=best_book.get("abstract"),
            matched_recommendation=recommendation_title,
            content_summary=None,
        )

    except Exception as e:
        logger.warning(f"Book search failed for '{query}': {e}")
        return None


async def search_books(state: dict) -> dict[str, Any]:
    """Search for all recommended books using book_search API.

    Collects recommendations from all three categories and searches
    for each one in parallel, prioritizing PDF format.
    """
    # Collect all recommendations from all categories
    all_recommendations = (
        state.get("analogous_recommendations", []) +
        state.get("inspiring_recommendations", []) +
        state.get("expressive_recommendations", [])
    )

    if not all_recommendations:
        logger.warning("No recommendations to search for")
        return {"search_results": []}

    # Extract language from state
    language_config = state.get("language_config")
    language = language_config.get("code") if language_config else None

    # Search all recommendations in parallel
    search_tasks = [
        _search_single_book(
            recommendation_title=rec["title"],
            author=rec.get("author"),
            language=language,
        )
        for rec in all_recommendations
    ]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Collect results and deduplicate
    results = []
    seen_md5 = set()

    for result in search_results:
        if isinstance(result, Exception):
            logger.warning(f"Search task failed: {result}")
            continue

        book = result
        if book:
            # Deduplicate by MD5
            if book["md5"] and book["md5"] in seen_md5:
                logger.debug(f"Skipping duplicate book: {book['title']}")
                continue

            if book["md5"]:
                seen_md5.add(book["md5"])

            results.append(book)

    logger.info(
        f"Found {len(results)} books from {len(all_recommendations)} recommendations"
    )

    return {"search_results": results}
