"""Playwright-based web scraper for fallback when Firecrawl fails."""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import html2text

if TYPE_CHECKING:
    from playwright.async_api import Browser, Download, Playwright

logger = logging.getLogger(__name__)


class PDFDownloadDetected(Exception):
    """Raised when a PDF download is detected instead of a web page.

    The PDF content is available in the `content` attribute.
    """

    def __init__(self, content: bytes, url: str):
        self.content = content
        self.url = url
        super().__init__(f"PDF download detected for {url} ({len(content)} bytes)")


class PlaywrightScraper:
    """Fallback scraper using headless browser.

    Features:
    - Lazy browser initialization (only starts when first needed)
    - Rate limiting between requests
    - Clean HTML to markdown conversion
    - Proper resource cleanup
    """

    def __init__(
        self,
        timeout: int | None = None,
        delay: float | None = None,
        headless: bool | None = None,
    ):
        """Initialize the Playwright scraper.

        Args:
            timeout: Page load timeout in milliseconds (default: 60000)
            delay: Delay between requests in seconds (default: 1.5)
            headless: Run browser in headless mode (default: True)
        """
        self._playwright: "Playwright | None" = None
        self._browser: "Browser | None" = None

        # Config from env or defaults
        self._timeout = timeout or int(
            os.environ.get("SCRAPER_PLAYWRIGHT_TIMEOUT", "60000")
        )
        self._delay = delay or float(os.environ.get("SCRAPER_PLAYWRIGHT_DELAY", "1.5"))
        self._headless = (
            headless
            if headless is not None
            else (
                os.environ.get("SCRAPER_PLAYWRIGHT_HEADLESS", "true").lower() == "true"
            )
        )

        self._last_request: float = 0

        # Configure html2text
        self._html2text = html2text.HTML2Text()
        self._html2text.ignore_links = False
        self._html2text.ignore_images = True
        self._html2text.ignore_emphasis = False
        self._html2text.body_width = 0  # Don't wrap lines

    async def _get_browser(self) -> "Browser":
        """Get or create browser instance (lazy initialization)."""
        if self._browser is None:
            from playwright.async_api import async_playwright

            logger.debug("Initializing Playwright browser")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )
            logger.info("Playwright browser started")
        return self._browser

    async def _rate_limit(self) -> None:
        """Enforce delay between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request
        if self._last_request > 0 and elapsed < self._delay:
            wait_time = self._delay - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        self._last_request = time.monotonic()

    async def scrape(self, url: str) -> str:
        """Scrape URL and return content as markdown.

        Args:
            url: The URL to scrape

        Returns:
            Page content converted to markdown

        Raises:
            PDFDownloadDetected: If the URL triggers a PDF download instead of a page.
                The PDF content is available in the exception's `content` attribute.
        """
        await self._rate_limit()

        browser = await self._get_browser()

        # Create context with realistic browser fingerprint
        # Enable downloads to handle PDF URLs gracefully
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            accept_downloads=True,
        )
        page = await context.new_page()
        download_content: bytes | None = None
        download_path: str | None = None

        async def handle_download(download: "Download") -> None:
            """Handle file downloads (e.g., PDF files)."""
            nonlocal download_content, download_path
            try:
                # Save download to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    download_path = tmp.name
                await download.save_as(download_path)
                download_content = Path(download_path).read_bytes()
                logger.debug(
                    f"Playwright captured download: {len(download_content)} bytes"
                )
            except Exception as e:
                logger.warning(f"Failed to capture download: {e}")

        page.on("download", handle_download)

        try:
            logger.debug(f"Playwright navigating to {url}")

            # Use domcontentloaded instead of networkidle for initial navigation
            # This allows us to detect downloads before they timeout
            try:
                await page.goto(
                    url,
                    timeout=self._timeout,
                    wait_until="domcontentloaded",
                )
            except Exception:
                # Check if we captured a download (PDF URL case)
                if download_content:
                    logger.debug(
                        "Navigation failed but download captured - likely PDF URL"
                    )
                    raise PDFDownloadDetected(download_content, url)
                raise

            # Give a moment for any downloads to start
            await asyncio.sleep(0.5)

            # Check if a download was triggered instead of a page
            if download_content:
                raise PDFDownloadDetected(download_content, url)

            # Wait for network to settle
            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=self._timeout // 2
                )
            except Exception:
                # If networkidle times out, proceed with what we have
                logger.debug("networkidle timeout, proceeding with current content")

            # Get page HTML
            html = await page.content()

            # Convert to markdown
            markdown = self._html2text.handle(html)

            logger.debug(f"Playwright scraped {len(markdown)} chars")
            return markdown.strip()

        except PDFDownloadDetected:
            raise

        except Exception as e:
            logger.error(f"Playwright scrape failed: {e}")
            raise

        finally:
            # Cleanup download temp file
            if download_path:
                try:
                    Path(download_path).unlink(missing_ok=True)
                except OSError:
                    pass
            await page.close()
            await context.close()

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            logger.debug("Closing Playwright browser")
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
            logger.debug("Playwright browser closed")

    async def __aenter__(self) -> "PlaywrightScraper":
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        await self.close()
