"""
Book search node with multi-query fallback strategy.

Searches for books matching the LLM recommendations using multiple
query strategies to maximize finding results.
"""

import asyncio
import logging
from typing import Any

from langchain_tools.book_search import book_search
from workflows.research.book_finding.state import BookResult

logger = logging.getLogger(__name__)

# Search configuration
MAX_RESULTS_PER_SEARCH = 15
MAX_CONCURRENT_SEARCHES = 5


async def _search_single_book(
    recommendation_title: str,
    author: str | None,
    language: str | None = None,
) -> BookResult | None:
    """Search for a single recommended book with fallback strategies.

    Tries multiple query strategies:
    1. Title + Author (if author known)
    2. Title only
    3. Author only (if author known)

    Args:
        recommendation_title: Title of the recommended book
        author: Optional author name to improve search
        language: Optional language filter (e.g., "en", "es")

    Returns:
        BookResult if found, None otherwise
    """
    # Build list of queries to try
    queries = []
    if author:
        queries.append(f"{recommendation_title} {author}")
    queries.append(recommendation_title)
    if author:
        queries.append(author)

    all_results = []

    for query in queries:
        try:
            search_params = {
                "query": query,
                "limit": MAX_RESULTS_PER_SEARCH,
            }
            if language:
                search_params["language"] = language

            result = await book_search.ainvoke(search_params)
            books = result.get("results", [])

            if books:
                logger.debug(f"Query '{query}' found {len(books)} results")
                all_results.extend(books)
                # If we found results with title+author, that's good enough
                if len(queries) > 1 and query == queries[0]:
                    break
            else:
                logger.debug(f"No results for query '{query}'")

        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            continue

    if not all_results:
        logger.warning(f"No results found for '{recommendation_title}'")
        return None

    # Deduplicate by MD5
    seen_md5 = set()
    unique_results = []
    for book in all_results:
        md5 = book.get("md5", "")
        if md5 and md5 not in seen_md5:
            seen_md5.add(md5)
            unique_results.append(book)

    # Prioritize PDF format
    pdf_books = [b for b in unique_results if b.get("format", "").lower() == "pdf"]
    best_book = pdf_books[0] if pdf_books else unique_results[0]

    logger.debug(f"Matched '{recommendation_title}' to '{best_book.get('title', 'Unknown')}'")

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

    logger.debug(f"Searching for {len(all_recommendations)} book recommendations (language: {language})")

    # Limit concurrency to avoid overwhelming the service
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def search_with_semaphore(rec: dict) -> BookResult | None:
        async with semaphore:
            return await _search_single_book(
                recommendation_title=rec["title"],
                author=rec.get("author"),
                language=language,
            )

    # Search all recommendations in parallel
    search_tasks = [search_with_semaphore(rec) for rec in all_recommendations]
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
        f"Book search complete: found {len(results)}/{len(all_recommendations)} books"
    )

    return {"search_results": results}
