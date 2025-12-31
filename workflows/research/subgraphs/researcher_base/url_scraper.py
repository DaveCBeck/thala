"""URL scraping utilities."""

import asyncio
import logging
from typing import Any

from langchain_tools.firecrawl import scrape_url
from workflows.research.state import ResearcherState, WebSearchResult

from .cache import _scrape_cache
from .pdf_processor import is_pdf_url, fetch_pdf_via_marker

logger = logging.getLogger(__name__)


async def scrape_single_url(
    result: WebSearchResult,
    index: int,
    max_content_length: int = 8000,
) -> tuple[int, str | None, WebSearchResult]:
    """Scrape a single URL and return the result.

    This helper function enables parallel scraping via asyncio.gather().
    Uses a TTL cache to avoid re-scraping the same URL across researchers or iterations.

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

    # Check cache first
    if url in _scrape_cache:
        content = _scrape_cache[url]
        logger.debug(f"Cache hit for: {url} ({len(content)} chars)")

        # Format scraped content string
        content_str = f"[{result['title']}]\nURL: {result['url']}\n\n{content}"
        updated_result["content"] = content

        return (index, content_str, updated_result)

    try:
        # Route PDFs to Marker instead of Firecrawl
        if is_pdf_url(url):
            logger.info(f"Processing PDF via Marker: {url}")
            content = await fetch_pdf_via_marker(url)
            if not content:
                # Fallback to Firecrawl if Marker fails
                logger.info(f"Marker failed, falling back to Firecrawl: {url}")
                response = await scrape_url.ainvoke({"url": url})
                content = response.get("markdown", "")
        else:
            response = await scrape_url.ainvoke({"url": url})
            content = response.get("markdown", "")

        # Truncate very long content
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n\n[Content truncated...]"

        # Store in cache for future use
        _scrape_cache[url] = content
        logger.debug(f"Scraped and cached {len(content)} chars from: {url}")

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

    Routes PDF URLs to local Marker service instead of Firecrawl to save API costs.
    Uses asyncio.gather() to scrape multiple URLs concurrently for improved performance.
    Results are cached with 1-hour TTL to avoid redundant scrapes across researchers.

    Args:
        state: The researcher state containing search results
        max_scrapes: Maximum number of URLs to scrape
    """
    results = state.get("search_results", [])

    if not results:
        return {"scraped_content": [], "search_results": []}

    # Check how many URLs are already cached
    urls_to_scrape = [r["url"] for r in results[:max_scrapes]]
    cached_count = sum(1 for url in urls_to_scrape if url in _scrape_cache)
    logger.info(
        f"Scraping {len(urls_to_scrape)} URLs ({cached_count} cached, "
        f"{len(urls_to_scrape) - cached_count} new) - cache size: {len(_scrape_cache)}"
    )

    # Create scraping tasks for parallel execution
    scraping_tasks = [
        scrape_single_url(result, i)
        for i, result in enumerate(results[:max_scrapes])
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

    logger.info(f"Scraped {len(scraped)} URLs successfully in parallel")

    return {
        "scraped_content": scraped,
        "search_results": updated_results,
    }
