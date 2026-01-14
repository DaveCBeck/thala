"""Unified scraping service with automatic fallback.

Fallback chain:
1. Local Firecrawl (self-hosted) - fast, no rate limits
2. Cloud Firecrawl (stealth) - for sites needing anti-bot bypass
3. Playwright (local browser) - for blocklisted/unreachable sites
"""

import asyncio
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .errors import LocalServiceUnavailableError, ScrapingError, SiteBlockedError
from .playwright_scraper import PDFDownloadDetected, PlaywrightScraper

if TYPE_CHECKING:
    from .firecrawl_clients import FirecrawlClients

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


def _is_local_unavailable_error(error: Exception) -> bool:
    """Check if error indicates local service is unavailable."""
    error_str = str(error).lower()

    unavailable_indicators = [
        "connection refused",
        "no route to host",
        "name resolution",
        "network is unreachable",
        "cannot connect",
    ]
    return any(ind in error_str for ind in unavailable_indicators)


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
            logger.debug(
                f"Transient error on attempt {attempt}/{MAX_RETRY_ATTEMPTS}, "
                f"retrying in {delay}s: {e}"
            )
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise last_error


class ScraperService:
    """Unified scraping service with automatic fallback.

    Fallback chain:
    1. Local Firecrawl (if configured) - fast, free, no rate limits
    2. Cloud Firecrawl stealth - for sites with anti-bot measures
    3. Playwright (local browser) - for blocklisted/unreachable sites

    Usage:
        service = ScraperService()
        result = await service.scrape("https://example.com")

        # Or as context manager for proper cleanup:
        async with ScraperService() as service:
            result = await service.scrape("https://example.com")
    """

    def __init__(self):
        """Initialize the scraper service."""
        self._firecrawl_clients: "FirecrawlClients | None" = None
        self._playwright: PlaywrightScraper | None = None
        self._blocklist: set[str] = set()  # Domains that require Playwright

    def _get_firecrawl_clients(self) -> "FirecrawlClients":
        """Get Firecrawl client manager (lazy initialization)."""
        if self._firecrawl_clients is None:
            from .firecrawl_clients import get_firecrawl_clients

            self._firecrawl_clients = get_firecrawl_clients()
            logger.debug("Firecrawl clients manager initialized")
        return self._firecrawl_clients

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

    async def _scrape_local(
        self, url: str, include_links: bool = False
    ) -> ScrapeResult:
        """Scrape using local (self-hosted) Firecrawl."""
        clients = self._get_firecrawl_clients()
        client = clients.local

        if client is None:
            raise LocalServiceUnavailableError(
                "Local Firecrawl not configured",
                url=url,
                provider="firecrawl-local",
            )

        formats = ["markdown"]
        if include_links:
            formats.append("links")

        try:
            result = await client.scrape(url, formats=formats)

            # Check for blocked response
            if self._is_blocked_response(result):
                raise SiteBlockedError(
                    "Response appears to be blocked",
                    url=url,
                    provider="firecrawl-local",
                )

            # Extract data from Document object
            markdown = result.markdown if hasattr(result, "markdown") else ""
            links = []
            if include_links and hasattr(result, "links"):
                links = result.links or []

            return ScrapeResult(
                url=url,
                markdown=markdown or "",
                links=links,
                provider="firecrawl-local",
            )

        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            # Local service unavailable - wrap in our error type
            if _is_local_unavailable_error(e):
                raise LocalServiceUnavailableError(
                    str(e), url=url, provider="firecrawl-local"
                )
            raise

    async def _scrape_cloud_stealth(
        self, url: str, include_links: bool = False
    ) -> ScrapeResult:
        """Scrape using cloud Firecrawl with stealth proxy."""
        from firecrawl.v2.utils.error_handler import WebsiteNotSupportedError

        clients = self._get_firecrawl_clients()
        client = clients.cloud

        if client is None:
            raise ValueError(
                "Cloud Firecrawl not configured (FIRECRAWL_API_KEY required)"
            )

        formats = ["markdown"]
        if include_links:
            formats.append("links")

        try:
            result = await client.scrape(url, formats=formats, proxy="stealth")

            # Check for blocked response
            if self._is_blocked_response(result):
                raise SiteBlockedError(
                    "Response appears to be blocked even with stealth",
                    url=url,
                    provider="firecrawl-stealth",
                )

            # Extract data from Document object
            markdown = result.markdown if hasattr(result, "markdown") else ""
            links = []
            if include_links and hasattr(result, "links"):
                links = result.links or []

            return ScrapeResult(
                url=url,
                markdown=markdown or "",
                links=links,
                provider="firecrawl-stealth",
            )

        except WebsiteNotSupportedError as e:
            # Site is explicitly blocked by Firecrawl
            raise SiteBlockedError(str(e), url=url, provider="firecrawl-stealth")

    async def _scrape_playwright(
        self, url: str, include_links: bool = False
    ) -> ScrapeResult:
        """Scrape using Playwright fallback.

        If the URL triggers a PDF download, the PDF is converted to markdown
        using the Marker service.
        """
        scraper = self._get_playwright()

        try:
            markdown = await scraper.scrape(url)
        except PDFDownloadDetected as e:
            # URL was a PDF - convert to markdown using Marker
            logger.info(
                f"Playwright detected PDF download ({len(e.content)} bytes), converting via Marker"
            )
            from .pdf import process_pdf_bytes

            try:
                markdown = await process_pdf_bytes(e.content)
                return ScrapeResult(
                    url=url,
                    markdown=markdown,
                    links=[],
                    provider="playwright-pdf",
                )
            except Exception as pdf_error:
                logger.warning(f"PDF conversion failed: {pdf_error}")
                raise ScrapingError(
                    f"PDF detected but conversion failed: {pdf_error}",
                    url=url,
                    provider="playwright-pdf",
                )

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
        1. Local Firecrawl (skip if not configured or domain in blocklist)
        2. Cloud Firecrawl stealth (on local failure or blocked response)
        3. Playwright (on cloud stealth failure or site blocked)

        Args:
            url: The URL to scrape
            include_links: Whether to extract links from the page

        Returns:
            ScrapeResult with markdown content and metadata
        """
        domain = _extract_domain(url)
        clients = self._get_firecrawl_clients()

        # Skip straight to Playwright for known-blocked domains
        if domain in self._blocklist:
            logger.debug(f"Domain {domain} in blocklist, using Playwright")
            return await self._scrape_playwright(url, include_links)

        # === Tier 1: Local Firecrawl ===
        if clients.config.local_available:
            try:
                logger.debug("Trying local Firecrawl")
                return await _with_retry(
                    self._scrape_local, url, include_links=include_links
                )

            except LocalServiceUnavailableError as e:
                # Local service down - proceed to cloud (don't add to blocklist)
                logger.warning(f"Local Firecrawl unavailable: {e}")

            except SiteBlockedError:
                # Site blocked locally - try cloud stealth
                logger.debug(
                    "Local Firecrawl got blocked response, trying cloud stealth"
                )

            except Exception as e:
                logger.debug(f"Local Firecrawl failed: {e}")

        # === Tier 2: Cloud Firecrawl Stealth ===
        if clients.config.cloud_available:
            try:
                logger.debug("Trying cloud Firecrawl stealth")
                return await _with_retry(
                    self._scrape_cloud_stealth, url, include_links=include_links
                )

            except SiteBlockedError:
                # Site blocked even with stealth - add to blocklist
                logger.warning(
                    f"Site {domain} blocked by cloud stealth, adding to blocklist"
                )
                self._blocklist.add(domain)

            except Exception as e:
                logger.debug(f"Cloud stealth failed: {e}")

        # === Tier 3: Playwright Fallback ===
        logger.debug("Falling back to Playwright")
        try:
            return await _with_retry(self._scrape_playwright, url, include_links)
        except Exception as e:
            logger.error(f"All scraping methods failed: {e}")
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


async def close_scraper_service() -> None:
    """Close the global scraper service and release resources."""
    global _scraper_service
    if _scraper_service is not None:
        await _scraper_service.close()
        _scraper_service = None


def get_scraper_service() -> ScraperService:
    """Get the global scraper service instance."""
    global _scraper_service
    if _scraper_service is None:
        from core.utils.async_http_client import register_cleanup

        _scraper_service = ScraperService()
        register_cleanup("Scraper", close_scraper_service)
    return _scraper_service
