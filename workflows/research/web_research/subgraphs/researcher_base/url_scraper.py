"""URL scraping utilities."""

import asyncio
import logging
from typing import Any

from core.scraping import get_url, GetUrlOptions
from workflows.research.web_research.state import ResearcherState, WebSearchResult

logger = logging.getLogger(__name__)


async def scrape_single_url(
    result: WebSearchResult,
    index: int,
    max_content_length: int = 8000,
) -> tuple[int, str | None, WebSearchResult]:
    """Scrape a single URL and return the result.

    This helper function enables parallel scraping via asyncio.gather().

    Args:
        result: WebSearchResult containing URL and metadata
        index: Original index in results list (for preserving order)
        max_content_length: Maximum content length before truncation

    Returns:
        Tuple of (index, scraped_content_str, updated_result)
        - index: Original position for deterministic ordering
        - scraped_content_str: Formatted content string or None on failure
        - updated_result: Result dict with content field added
    """
    url = result["url"]
    updated_result = dict(result)
    updated_result["_index"] = index

    try:
        # Use unified get_url() - handles PDFs, HTML, fallbacks automatically
        url_result = await get_url(
            url,
            GetUrlOptions(
                detect_academic=False,  # Web research doesn't need classification
                allow_retrieve_academic=False,
            ),
        )
        content = url_result.content

        # Truncate very long content
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[Content truncated...]"

        logger.debug(f"Scraped {len(content)} chars from: {url}")

        # Format scraped content string
        content_str = f"[{result['title']}]\nURL: {result['url']}\n\n{content}"

        # Update result with content
        updated_result["content"] = content

        return (index, content_str, updated_result)

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return (index, None, updated_result)


async def scrape_pages(
    state: ResearcherState,
    max_scrapes: int,
) -> dict[str, Any]:
    """Scrape top results for full content in parallel.

    Uses asyncio.gather() to scrape multiple URLs concurrently for improved performance.

    Args:
        state: The researcher state containing search results
        max_scrapes: Maximum number of URLs to scrape
    """
    results = state.get("search_results", [])

    if not results:
        return {"scraped_content": [], "search_results": []}

    urls_to_scrape = [r["url"] for r in results[:max_scrapes]]
    logger.debug(f"Scraping {len(urls_to_scrape)} URLs in parallel")

    # Create scraping tasks for parallel execution
    scraping_tasks = [
        scrape_single_url(result, i) for i, result in enumerate(results[:max_scrapes])
    ]

    # Execute all scrapes concurrently
    task_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

    # Process results
    scraped = []
    updated_results = []

    for task_result in task_results:
        if isinstance(task_result, Exception):
            logger.error(f"Scraping task failed with exception: {task_result}")
            continue

        idx, content_str, updated_result = task_result

        if content_str:
            scraped.append(content_str)

        updated_results.append(updated_result)

    # Sort by original index to maintain deterministic order
    updated_results.sort(key=lambda x: x.get("_index", 0))

    # Remove temporary index field
    for r in updated_results:
        r.pop("_index", None)

    # Keep remaining results without scraping
    updated_results.extend(results[max_scrapes:])

    logger.debug(f"Scraped {len(scraped)} URLs successfully")

    return {
        "scraped_content": scraped,
        "search_results": updated_results,
    }
