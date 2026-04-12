"""URL scraping utilities."""

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urlparse

from core.scraping import get_url, GetUrlOptions
from workflows.research.web_research.state import ResearcherState, WebSearchResult

logger = logging.getLogger(__name__)

# File extensions that are never useful web pages to scrape
_JUNK_EXTENSIONS = frozenset({
    ".txt", ".csv", ".tsv", ".json", ".xml", ".gz", ".zip", ".tar",
    ".bz2", ".xz", ".7z", ".rar", ".parquet", ".feather", ".arrow",
    ".sqlite", ".db", ".sql", ".log", ".dat", ".bin", ".exe",
    ".whl", ".egg", ".rpm", ".deb", ".xls", ".xlsx",
})

# URL path patterns indicating raw data / archive indexes rather than articles
_JUNK_PATH_PATTERNS = re.compile(
    r"(?:"
    r"/download/"
    r"|/raw/"
    r"|/archive/?$"
    r"|/dataset[s]?/"
    r"|/bulk/"
    r"|/dump[s]?/"
    r")",
    re.IGNORECASE,
)


def is_scrapable_url(url: str) -> bool:
    """Return False for URLs that are obviously not web pages worth scraping."""
    try:
        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        for ext in _JUNK_EXTENSIONS:
            if path_lower.endswith(ext):
                return False

        if _JUNK_PATH_PATTERNS.search(parsed.path):
            return False

        return True
    except Exception:
        return True  # Fail open on parse errors


async def scrape_single_url(
    result: WebSearchResult,
    index: int,
) -> tuple[int, str | None, WebSearchResult]:
    """Scrape a single URL and return the result.

    This helper function enables parallel scraping via asyncio.gather().

    Args:
        result: WebSearchResult containing URL and metadata
        index: Original index in results list (for preserving order)

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
    scraping_tasks = [scrape_single_url(result, i) for i, result in enumerate(results[:max_scrapes])]

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
