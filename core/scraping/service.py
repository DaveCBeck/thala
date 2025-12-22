"""Unified scraping service with automatic fallback.

Fallback chain:
1. Firecrawl (basic) - fast, works for most sites
2. Firecrawl (stealth) - for sites with anti-bot measures
3. Playwright (local browser) - for blocklisted/unreachable sites
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .errors import ScrapingError, SiteBlockedError
from .playwright_scraper import PlaywrightScraper

if TYPE_CHECKING:
    from firecrawl import AsyncFirecrawl

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 2
RETRY_INITIAL_DELAY = 2.0


class ScrapeResult(BaseModel):
    """Output schema for scrape operations."""

    url: str
    markdown: str
    links: list[str] = Field(default_factory=list)
    provider: str = "unknown"  # Which provider succeeded


def _extract_domain(url: str) -> str:
    """Extract domain from URL for blocklist matching."""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def _is_transient_error(error: Exception) -> bool:
    """Check if error is transient and should be retried."""
    error_str = str(error).lower()
    error_type = type(error).__name__

    # HTTP status codes that are transient
    transient_statuses = ["502", "503", "504"]
    if any(status in error_str for status in transient_statuses):
        return True

    # Connection/timeout errors
    transient_indicators = [
        "timeout",
        "timed out",
        "connection reset",
        "connection refused",
        "temporary failure",
    ]
    if any(indicator in error_str for indicator in transient_indicators):
        return True

    # Common exception types that are transient
    transient_types = [
        "TimeoutError",
        "ClientConnectorError",
        "ServerTimeoutError",
        "asyncio.TimeoutError",
    ]
    if any(t in error_type for t in transient_types):
        return True

    return False


async def _with_retry(func, *args, **kwargs):
    """Execute function with retry logic for transient failures."""
    last_error = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            # Don't retry permanent errors
            if not _is_transient_error(e):
                raise

            # Don't retry on last attempt
            if attempt >= MAX_RETRY_ATTEMPTS:
                raise

            # Calculate exponential backoff delay
            delay = RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
            logger.info(
                f"Transient error on attempt {attempt}/{MAX_RETRY_ATTEMPTS}, "
                f"retrying in {delay}s: {e}"
            )
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise last_error


class ScraperService:
    """Unified scraping service with automatic fallback.

    Usage:
        service = ScraperService()
        result = await service.scrape("https://example.com")

        # Or as context manager for proper cleanup:
        async with ScraperService() as service:
            result = await service.scrape("https://example.com")
    """

    def __init__(self):
        """Initialize the scraper service."""
        self._firecrawl: "AsyncFirecrawl | None" = None
        self._playwright: PlaywrightScraper | None = None
        self._blocklist: set[str] = set()  # Domains that require Playwright

    def _get_firecrawl(self) -> "AsyncFirecrawl":
        """Get or create Firecrawl client (lazy initialization)."""
        if self._firecrawl is None:
            from firecrawl import AsyncFirecrawl

            api_key = os.environ.get("FIRECRAWL_API_KEY")
            if not api_key:
                raise ValueError(
                    "FIRECRAWL_API_KEY environment variable is required. "
                    "Get one at https://firecrawl.dev"
                )
            self._firecrawl = AsyncFirecrawl(api_key=api_key)
            logger.debug("Firecrawl client initialized")
        return self._firecrawl

    def _get_playwright(self) -> PlaywrightScraper:
        """Get or create Playwright scraper (lazy initialization)."""
        if self._playwright is None:
            self._playwright = PlaywrightScraper()
            logger.debug("Playwright scraper initialized")
        return self._playwright

    def _is_blocked_response(self, result) -> bool:
        """Check if response indicates site blocking (e.g., captcha, login wall)."""
        if not hasattr(result, "markdown"):
            return True

        markdown = result.markdown or ""

        # Check for common blocking indicators
        if len(markdown) < 100:
            return True

        blocking_indicators = [
            "captcha",
            "access denied",
            "please verify",
            "bot detection",
            "enable javascript",
        ]
        markdown_lower = markdown.lower()
        return any(indicator in markdown_lower for indicator in blocking_indicators)

    async def _scrape_firecrawl(
        self, url: str, proxy: str | None = None, include_links: bool = False
    ) -> ScrapeResult:
        """Scrape using Firecrawl."""
        from firecrawl.v2.utils.error_handler import WebsiteNotSupportedError

        client = self._get_firecrawl()
        formats = ["markdown"]
        if include_links:
            formats.append("links")

        try:
            if proxy:
                result = await client.scrape(url, formats=formats, proxy=proxy)
            else:
                result = await client.scrape(url, formats=formats)

            # Check for blocked response
            if self._is_blocked_response(result):
                raise SiteBlockedError(
                    "Response appears to be blocked", url=url, provider="firecrawl"
                )

            # Extract data from Document object
            markdown = result.markdown if hasattr(result, "markdown") else ""
            links = []
            if include_links and hasattr(result, "links"):
                links = result.links or []

            provider = f"firecrawl-{proxy}" if proxy else "firecrawl"
            return ScrapeResult(
                url=url,
                markdown=markdown or "",
                links=links,
                provider=provider,
            )

        except WebsiteNotSupportedError as e:
            # Site is explicitly blocked by Firecrawl
            raise SiteBlockedError(str(e), url=url, provider="firecrawl")

    async def _scrape_playwright(
        self, url: str, include_links: bool = False
    ) -> ScrapeResult:
        """Scrape using Playwright fallback."""
        scraper = self._get_playwright()
        markdown = await scraper.scrape(url)

        # TODO: Extract links from markdown if include_links is True
        # For now, we don't extract links from Playwright results
        return ScrapeResult(
            url=url,
            markdown=markdown,
            links=[],
            provider="playwright",
        )

    async def scrape(self, url: str, include_links: bool = False) -> ScrapeResult:
        """Scrape URL with automatic fallback chain.

        Fallback order:
        1. Firecrawl basic (skip if domain in blocklist)
        2. Firecrawl stealth (on failure or blocked response)
        3. Playwright (on WebsiteNotSupportedError or stealth failure)

        Args:
            url: The URL to scrape
            include_links: Whether to extract links from the page

        Returns:
            ScrapeResult with markdown content and metadata
        """
        domain = _extract_domain(url)

        # Skip straight to Playwright for known-blocked domains
        if domain in self._blocklist:
            logger.debug(f"Domain {domain} in blocklist, using Playwright directly")
            return await self._scrape_playwright(url, include_links)

        # Try Firecrawl basic with retry
        try:
            logger.debug(f"Trying Firecrawl basic for {url}")
            return await _with_retry(
                self._scrape_firecrawl, url, proxy=None, include_links=include_links
            )

        except SiteBlockedError as e:
            if "WebsiteNotSupportedError" in str(type(e).__mro__) or "not supported" in str(e).lower():
                # Firecrawl explicitly blocks this site
                logger.info(f"Site {domain} blocked by Firecrawl, adding to blocklist")
                self._blocklist.add(domain)
                return await self._scrape_playwright(url, include_links)
            # Response blocked but site not in Firecrawl blocklist - try stealth
            logger.debug(f"Blocked response from {url}, trying stealth")

        except Exception as e:
            logger.debug(f"Firecrawl basic failed for {url}: {e}")

        # Try Firecrawl stealth with retry
        try:
            logger.debug(f"Trying Firecrawl stealth for {url}")
            return await _with_retry(
                self._scrape_firecrawl, url, proxy="stealth", include_links=include_links
            )

        except SiteBlockedError as e:
            # Site blocked even with stealth - add to blocklist
            logger.info(f"Site {domain} blocked by Firecrawl stealth, adding to blocklist")
            self._blocklist.add(domain)

        except Exception as e:
            logger.debug(f"Firecrawl stealth failed for {url}: {e}")

        # Playwright fallback with retry
        logger.debug(f"Falling back to Playwright for {url}")
        try:
            return await _with_retry(self._scrape_playwright, url, include_links)
        except Exception as e:
            logger.error(f"All scraping methods failed for {url}: {e}")
            raise ScrapingError(
                f"All scraping methods failed: {e}",
                url=url,
                provider="all",
            )

    async def close(self) -> None:
        """Clean up all resources."""
        if self._playwright:
            await self._playwright.close()
            self._playwright = None

    async def __aenter__(self) -> "ScraperService":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        await self.close()


# Module-level singleton for convenience
_scraper_service: ScraperService | None = None


def get_scraper_service() -> ScraperService:
    """Get the global scraper service instance."""
    global _scraper_service
    if _scraper_service is None:
        _scraper_service = ScraperService()
    return _scraper_service


async def close_scraper_service() -> None:
    """Close the global scraper service and release resources."""
    global _scraper_service
    if _scraper_service is not None:
        await _scraper_service.close()
        _scraper_service = None
