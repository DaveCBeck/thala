"""
Firecrawl web research tools for LangChain.

Provides: web_search, scrape_url, map_website
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client Management (lazy singleton)
# ---------------------------------------------------------------------------

_firecrawl_client = None


def _get_firecrawl():
    """Get AsyncFirecrawl client (lazy init)."""
    global _firecrawl_client
    if _firecrawl_client is None:
        from firecrawl import AsyncFirecrawl

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY environment variable is required. "
                "Get one at https://firecrawl.dev"
            )
        _firecrawl_client = AsyncFirecrawl(api_key=api_key)
    return _firecrawl_client


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------


class WebSearchResult(BaseModel):
    """Individual search result."""

    title: str
    url: str
    description: Optional[str] = None


class WebSearchOutput(BaseModel):
    """Output schema for web_search tool."""

    query: str
    total_results: int
    results: list[WebSearchResult]


class ScrapeOutput(BaseModel):
    """Output schema for scrape_url tool."""

    url: str
    markdown: str
    links: list[str] = Field(default_factory=list)


class MapOutput(BaseModel):
    """Output schema for map_website tool."""

    url: str
    total_urls: int
    urls: list[str]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def web_search(query: str, limit: int = 5) -> dict:
    """Search the web for information on a topic.

    Use this when you need current information from the internet that may not
    be in your training data or memory stores.

    Args:
        query: What to search for on the web
        limit: Maximum number of results to return (default 5, max 20)

    Returns:
        Search results with titles, URLs, and descriptions.
    """
    client = _get_firecrawl()
    limit = min(max(1, limit), 20)

    try:
        response = await client.search(query, limit=limit)

        results = []
        web_results = response.get("data", []) if isinstance(response, dict) else []

        for item in web_results:
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description"),
                )
            )

        output = WebSearchOutput(
            query=query,
            total_results=len(results),
            results=results,
        )
        logger.debug(f"web_search returned {len(results)} results for: {query}")
        return output.model_dump(mode="json")

    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return WebSearchOutput(
            query=query,
            total_results=0,
            results=[],
        ).model_dump(mode="json")


@tool
async def scrape_url(url: str, include_links: bool = False) -> dict:
    """Scrape a webpage and return its content as markdown.

    Use this to read the full content of a specific URL. Works on most websites
    including those with JavaScript-rendered content.

    Args:
        url: The URL to scrape
        include_links: Whether to extract links from the page (default False)

    Returns:
        The page content as markdown, plus optionally a list of links found.
    """
    client = _get_firecrawl()

    formats = ["markdown"]
    if include_links:
        formats.append("links")

    try:
        response = await client.scrape(url, formats=formats)

        markdown = response.get("markdown", "") if isinstance(response, dict) else ""
        links = []
        if include_links:
            links = response.get("links", []) if isinstance(response, dict) else []

        output = ScrapeOutput(
            url=url,
            markdown=markdown,
            links=links,
        )
        logger.debug(f"scrape_url got {len(markdown)} chars from: {url}")
        return output.model_dump(mode="json")

    except Exception as e:
        logger.error(f"scrape_url failed for {url}: {e}")
        return ScrapeOutput(
            url=url,
            markdown=f"Error scraping URL: {e}",
            links=[],
        ).model_dump(mode="json")


@tool
async def map_website(url: str, limit: int = 50) -> dict:
    """Discover all URLs on a website.

    Use this to explore a website's structure before deciding which specific
    pages to scrape. Returns a list of discovered URLs.

    Args:
        url: The website URL to map (e.g., "https://example.com")
        limit: Maximum number of URLs to discover (default 50, max 500)

    Returns:
        List of URLs found on the website.
    """
    client = _get_firecrawl()
    limit = min(max(1, limit), 500)

    try:
        response = await client.map(url, limit=limit)

        urls = []
        if isinstance(response, dict):
            urls = response.get("links", []) or response.get("urls", [])
        elif isinstance(response, list):
            urls = response

        output = MapOutput(
            url=url,
            total_urls=len(urls),
            urls=urls[:limit],
        )
        logger.debug(f"map_website found {len(urls)} URLs on: {url}")
        return output.model_dump(mode="json")

    except Exception as e:
        logger.error(f"map_website failed for {url}: {e}")
        return MapOutput(
            url=url,
            total_urls=0,
            urls=[],
        ).model_dump(mode="json")
