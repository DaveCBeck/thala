"""PDF processing via Marker service.

Provides functions to convert PDFs to markdown using the Marker service.
Includes Playwright fallback for sites that block direct downloads.
"""

import asyncio
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from .detector import validate_pdf_bytes

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright

logger = logging.getLogger(__name__)

# Marker service configuration
MARKER_BASE_URL = os.getenv("MARKER_BASE_URL", "http://localhost:8001")
MARKER_INPUT_DIR = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
MARKER_POLL_INTERVAL = float(os.getenv("MARKER_POLL_INTERVAL", "2.0"))

# Module-level Playwright instances for reuse
_playwright: "Playwright | None" = None
_browser: "Browser | None" = None


class MarkerProcessingError(Exception):
    """Error during Marker PDF processing."""

    pass


async def _download_pdf_httpx(url: str, timeout: float = 60.0) -> bytes:
    """Download PDF from URL using httpx (simple/fast method).

    Args:
        url: URL to download from
        timeout: Request timeout in seconds

    Returns:
        PDF content bytes

    Raises:
        MarkerProcessingError: If download fails or content is not a valid PDF
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            content = response.content

            if not validate_pdf_bytes(content):
                raise MarkerProcessingError(
                    f"Downloaded content is not a valid PDF"
                )

            return content

        except httpx.HTTPStatusError as e:
            raise MarkerProcessingError(
                f"HTTP error downloading PDF: {e.response.status_code}"
            ) from e
        except httpx.TimeoutException as e:
            raise MarkerProcessingError(f"Timeout downloading PDF") from e
        except MarkerProcessingError:
            raise
        except Exception as e:
            raise MarkerProcessingError(f"Failed to download PDF: {e}") from e


async def _get_browser() -> "Browser":
    """Get or create browser instance (lazy initialization)."""
    global _playwright, _browser

    if _browser is None:
        from playwright.async_api import async_playwright

        logger.debug("Initializing Playwright browser for PDF download")
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        logger.info("Playwright browser started for PDF downloads")
    return _browser


async def _download_pdf_playwright(url: str, timeout: float = 60.0) -> bytes:
    """Download PDF using Playwright browser (for sites that block direct downloads).

    Uses a real browser context to bypass anti-bot measures that redirect
    httpx requests to login pages.

    Args:
        url: URL to download from
        timeout: Page load timeout in seconds

    Returns:
        PDF content bytes

    Raises:
        MarkerProcessingError: If download fails or content is not a valid PDF
    """
    browser = await _get_browser()

    # Create context with realistic browser fingerprint
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
        accept_downloads=True,
    )

    page = await context.new_page()
    content: bytes | None = None

    try:
        # Set up response interception to capture PDF bytes
        async def handle_response(response):
            nonlocal content
            content_type = response.headers.get("content-type", "")
            if "application/pdf" in content_type:
                try:
                    content = await response.body()
                    logger.debug(f"Captured PDF response: {len(content)} bytes")
                except Exception as e:
                    logger.debug(f"Failed to capture PDF body: {e}")

        page.on("response", handle_response)

        # Navigate to the PDF URL
        logger.debug(f"Playwright navigating to PDF URL")
        response = await page.goto(url, timeout=int(timeout * 1000), wait_until="networkidle")

        # Check if we captured the PDF via response interception
        if content and validate_pdf_bytes(content):
            logger.debug(f"Playwright captured PDF via response: {len(content) / 1024:.1f} KB")
            return content

        # If not captured via response, check if browser downloaded it
        # or if we can get it from the response directly
        if response:
            content_type = response.headers.get("content-type", "")
            if "application/pdf" in content_type:
                content = await response.body()
                if validate_pdf_bytes(content):
                    logger.debug(f"Playwright got PDF from response: {len(content) / 1024:.1f} KB")
                    return content

        raise MarkerProcessingError(
            f"Playwright could not download PDF (got non-PDF content)"
        )

    except MarkerProcessingError:
        raise
    except Exception as e:
        raise MarkerProcessingError(f"Playwright PDF download failed: {e}") from e
    finally:
        await page.close()
        await context.close()


async def _download_pdf(url: str, timeout: float = 60.0) -> bytes:
    """Download PDF from URL, trying httpx first then Playwright fallback.

    Args:
        url: URL to download from
        timeout: Request timeout in seconds

    Returns:
        PDF content bytes

    Raises:
        MarkerProcessingError: If all download methods fail
    """
    # Try simple httpx download first (fast)
    try:
        return await _download_pdf_httpx(url, timeout)
    except MarkerProcessingError as e:
        if "not a valid PDF" in str(e):
            # Site likely returned HTML (login page) - try Playwright
            logger.debug(f"httpx got non-PDF response, trying Playwright")
        else:
            raise

    # Fallback to Playwright (handles anti-bot redirects)
    return await _download_pdf_playwright(url, timeout)


def _save_to_marker_input(content: bytes, filename: Optional[str] = None) -> str:
    """Save PDF content to Marker input directory.

    Args:
        content: PDF content bytes
        filename: Optional filename (generates hash-based name if not provided)

    Returns:
        Path relative to Marker input directory (for API calls)
    """
    MARKER_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        # Generate filename from content hash
        content_hash = hashlib.md5(content).hexdigest()[:12]
        filename = f"pdf_{content_hash}.pdf"

    file_path = MARKER_INPUT_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)

    logger.debug(f"Saved PDF to Marker input: {file_path}")
    return filename


async def _submit_marker_job(
    file_path: str,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
) -> str:
    """Submit a PDF conversion job to Marker.

    Args:
        file_path: Path relative to Marker input directory
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR

    Returns:
        Job ID for polling
    """
    async with httpx.AsyncClient(base_url=MARKER_BASE_URL, timeout=30.0) as client:
        payload = {
            "file_path": file_path,
            "quality": quality,
            "markdown_only": False,
            "langs": langs or ["English"],
        }

        response = await client.post("/convert", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["job_id"]


async def _poll_marker_job(
    job_id: str,
    max_wait: Optional[float] = None,
) -> str:
    """Poll Marker job until completion.

    Args:
        job_id: Job ID to poll
        max_wait: Maximum wait time in seconds (None = no limit)

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If job fails or times out
    """
    start_time = asyncio.get_event_loop().time()

    async with httpx.AsyncClient(base_url=MARKER_BASE_URL, timeout=30.0) as client:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if max_wait is not None and elapsed > max_wait:
                raise MarkerProcessingError(
                    f"Marker job {job_id} did not complete within {max_wait}s"
                )

            response = await client.get(f"/jobs/{job_id}")
            response.raise_for_status()

            data = response.json()
            status = data["status"]

            if status == "completed":
                result = data["result"]
                return result.get("markdown", "")
            elif status == "failed":
                error = data.get("error", "Unknown error")
                raise MarkerProcessingError(f"Marker job {job_id} failed: {error}")
            elif status in ("pending", "processing"):
                await asyncio.sleep(MARKER_POLL_INTERVAL)
            else:
                raise MarkerProcessingError(f"Unknown Marker job status: {status}")


async def process_pdf_url(
    url: str,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: float = 120.0,
) -> str:
    """Download PDF from URL and convert to markdown via Marker.

    Args:
        url: URL to PDF file
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        timeout: Maximum time for entire operation

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If download or processing fails
    """
    logger.debug(f"Processing PDF URL")

    # Download PDF
    content = await _download_pdf(url, timeout=min(timeout, 60.0))
    logger.debug(f"Downloaded PDF: {len(content) / 1024:.1f} KB")

    # Process via Marker
    return await process_pdf_bytes(
        content,
        quality=quality,
        langs=langs,
        timeout=timeout,
    )


async def process_pdf_bytes(
    content: bytes,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: float = 120.0,
    filename: Optional[str] = None,
) -> str:
    """Convert PDF bytes to markdown via Marker.

    Args:
        content: PDF content bytes
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        timeout: Maximum time for processing
        filename: Optional filename for saved file

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If processing fails
    """
    if not validate_pdf_bytes(content):
        raise MarkerProcessingError("Content is not a valid PDF")

    # Save to Marker input directory
    file_path = _save_to_marker_input(content, filename)

    # Submit job
    job_id = await _submit_marker_job(file_path, quality=quality, langs=langs)
    logger.debug(f"Submitted Marker job: {job_id}")

    # Poll for completion
    markdown = await _poll_marker_job(job_id, max_wait=timeout)
    logger.debug(f"Marker conversion complete: {len(markdown)} chars")

    return markdown


async def process_pdf_file(
    file_path: str,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: float = 120.0,
) -> str:
    """Convert PDF file to markdown via Marker.

    Args:
        file_path: Path to PDF file
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        timeout: Maximum time for processing

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If processing fails
    """
    path = Path(file_path)
    if not path.exists():
        raise MarkerProcessingError(f"PDF file not found: {file_path}")

    with open(path, "rb") as f:
        content = f.read()

    return await process_pdf_bytes(
        content,
        quality=quality,
        langs=langs,
        timeout=timeout,
        filename=path.name,
    )
