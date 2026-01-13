"""
Firecrawl web research tools for LangChain.

Provides: web_search, scrape_url, map_website

Uses local (self-hosted) Firecrawl when available, with cloud fallback.
"""

import logging
import os
from typing import TYPE_CHECKING, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from .utils import clamp_limit, output_dict

if TYPE_CHECKING:
    from core.scraping.firecrawl_clients import FirecrawlClients

logger = logging.getLogger(__name__)

FIRECRAWL_TIMEOUT = int(os.environ.get("FIRECRAWL_TIMEOUT", "45"))


# ---------------------------------------------------------------------------
# Client Management (uses shared FirecrawlClients)
# ---------------------------------------------------------------------------

_firecrawl_clients: "FirecrawlClients | None" = None


def _get_clients() -> "FirecrawlClients":
    """Get FirecrawlClients singleton for tools."""
    global _firecrawl_clients
    if _firecrawl_clients is None:
        from core.scraping.firecrawl_clients import get_firecrawl_clients

        _firecrawl_clients = get_firecrawl_clients()
    return _firecrawl_clients


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
# Response Parsing Helpers
# ---------------------------------------------------------------------------


def _parse_search_response(response) -> list[WebSearchResult]:
    """Parse search response from either local or cloud Firecrawl."""
    results = []

    # Handle SearchData object from firecrawl v2
    if hasattr(response, "web") and response.web:
        for item in response.web:
            results.append(
                WebSearchResult(
                    title=getattr(item, "title", "") or "",
                    url=getattr(item, "url", "") or "",
                    description=getattr(item, "description", None),
                )
            )
    # Fallback for dict response (older API)
    elif isinstance(response, dict):
        web_results = response.get("data", [])
        for item in web_results:
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description"),
                )
            )

    return results


def _parse_map_response(response) -> list[str]:
    """Parse map response from either local or cloud Firecrawl."""
    urls = []

    # Handle MapData object from firecrawl v2
    if hasattr(response, "links") and response.links:
        urls = [item.url for item in response.links if hasattr(item, "url")]
    # Fallback for dict response (older API)
    elif isinstance(response, dict):
        urls = response.get("links", []) or response.get("urls", [])
    elif isinstance(response, list):
        urls = response

    return urls


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def web_search(
    query: str,
    limit: int = 5,
    locale: Optional[str] = None,
    preferred_domains: Optional[list[str]] = None,
) -> dict:
    """Search the web for information on a topic.

    Use this when you need current information from the internet that may not
    be in your training data or memory stores.

    Uses local Firecrawl (with SearXNG backend) when available,
    falls back to cloud Firecrawl if local fails.

    Args:
        query: What to search for on the web
        limit: Maximum number of results to return (default 5, max 20)
        locale: Language/region code (e.g., "es-ES", "zh-CN", "ja-JP")
        preferred_domains: Domain suffixes to prefer (e.g., [".es", ".mx"])

    Returns:
        Search results with titles, URLs, and descriptions.
    """
    clients = _get_clients()
    limit = clamp_limit(limit, min_val=1, max_val=20)

    # Build search params with locale hints
    search_params = {}
    if locale:
        search_params["locale"] = locale
    if preferred_domains:
        search_params["preferred_domains"] = preferred_domains

    # Try local first
    if clients.local:
        try:
            response = await clients.local.search(query, limit=limit, **search_params)
            results = _parse_search_response(response)
            if results:  # Only return if we got results
                logger.debug(
                    f"web_search (local) returned {len(results)} results for: {query}"
                )
                return output_dict(
                    WebSearchOutput(
                        query=query,
                        total_results=len(results),
                        results=results,
                    )
                )
            logger.debug("Local search returned no results, trying cloud")
        except Exception as e:
            logger.debug(f"Local search failed: {e}, trying cloud")

    # Fallback to cloud
    if clients.cloud:
        try:
            response = await clients.cloud.search(query, limit=limit, **search_params)
            results = _parse_search_response(response)
            logger.debug(
                f"web_search (cloud) returned {len(results)} results for: {query}"
            )
            return output_dict(
                WebSearchOutput(
                    query=query,
                    total_results=len(results),
                    results=results,
                )
            )
        except Exception as e:
            logger.error(f"Cloud search failed: {e}")

    # Both failed or not configured
    logger.warning(f"web_search failed for query: {query}")
    return output_dict(WebSearchOutput(query=query, total_results=0, results=[]))


@tool
async def scrape_url(url: str, include_links: bool = False) -> dict:
    """Scrape a webpage and return its content as markdown.

    Use this to read the full content of a specific URL. Works on most websites
    including those with JavaScript-rendered content.

    Uses automatic fallback: Local Firecrawl -> Cloud Firecrawl stealth -> Playwright.

    Args:
        url: The URL to scrape
        include_links: Whether to extract links from the page (default False)

    Returns:
        The page content as markdown, plus optionally a list of links found.
    """
    from core.scraping import get_scraper_service

    service = get_scraper_service()

    try:
        result = await service.scrape(url, include_links=include_links)

        output = ScrapeOutput(
            url=result.url,
            markdown=result.markdown,
            links=result.links,
        )
        logger.debug(
            f"scrape_url got {len(result.markdown)} chars from {url} "
            f"(provider: {result.provider})"
        )
        return output_dict(output)

    except Exception as e:
        logger.error(f"scrape_url failed for {url}: {e}")
        return output_dict(
            ScrapeOutput(
                url=url,
                markdown=f"Error scraping URL: {e}",
                links=[],
            )
        )


@tool
async def map_website(url: str, limit: int = 50) -> dict:
    """Discover all URLs on a website.

    Use this to explore a website's structure before deciding which specific
    pages to scrape. Returns a list of discovered URLs.

    Uses local Firecrawl when available, falls back to cloud.

    Args:
        url: The website URL to map (e.g., "https://example.com")
        limit: Maximum number of URLs to discover (default 50, max 500)

    Returns:
        List of URLs found on the website.
    """
    clients = _get_clients()
    limit = clamp_limit(limit, min_val=1, max_val=500)

    # Try local first
    if clients.local:
        try:
            response = await clients.local.map(url, limit=limit)
            urls = _parse_map_response(response)
            if urls:  # Only return if we got URLs
                logger.debug(f"map_website (local) found {len(urls)} URLs on: {url}")
                return output_dict(
                    MapOutput(
                        url=url,
                        total_urls=len(urls),
                        urls=urls[:limit],
                    )
                )
            logger.debug("Local map returned no URLs, trying cloud")
        except Exception as e:
            logger.debug(f"Local map failed: {e}, trying cloud")

    # Fallback to cloud
    if clients.cloud:
        try:
            response = await clients.cloud.map(url, limit=limit)
            urls = _parse_map_response(response)
            logger.debug(f"map_website (cloud) found {len(urls)} URLs on: {url}")
            return output_dict(
                MapOutput(
                    url=url,
                    total_urls=len(urls),
                    urls=urls[:limit],
                )
            )
        except Exception as e:
            logger.error(f"Cloud map failed: {e}")

    # Both failed or not configured
    logger.warning(f"map_website failed for URL: {url}")
    return output_dict(MapOutput(url=url, total_urls=0, urls=[]))
