"""PDF processing via Marker service.

Provides functions to convert PDFs to markdown using the Marker service.
Includes Playwright fallback for sites that block direct downloads.
Supports automatic chunking for large PDFs to prevent memory exhaustion.
"""

import asyncio
import gc
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from utils.pdf_chunking import (
    assemble_markdown_chunks,
    should_chunk_pdf,
    split_pdf_by_pages,
)

from .detector import validate_pdf_bytes

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright

logger = logging.getLogger(__name__)

# Marker service configuration
MARKER_BASE_URL = os.getenv("MARKER_BASE_URL", "http://localhost:8001")
MARKER_INPUT_DIR = Path(os.getenv("MARKER_INPUT_DIR", "/data/input"))
MARKER_POLL_INTERVAL = float(os.getenv("MARKER_POLL_INTERVAL", "15.0"))
MARKER_MAX_FILE_SIZE = int(os.getenv("MARKER_MAX_FILE_SIZE", str(1024 * 1024 * 1024)))  # 1GB

# Chunking configuration for large PDFs
MARKER_CHUNK_PAGE_THRESHOLD = int(os.getenv("MARKER_CHUNK_PAGE_THRESHOLD", "100"))
MARKER_CHUNK_SIZE = int(os.getenv("MARKER_CHUNK_SIZE", "100"))

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
                raise MarkerProcessingError("Downloaded content is not a valid PDF")

            return content

        except httpx.HTTPStatusError as e:
            raise MarkerProcessingError(
                f"HTTP error downloading PDF: {e.response.status_code}"
            ) from e
        except httpx.TimeoutException as e:
            raise MarkerProcessingError("Timeout downloading PDF") from e
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
    httpx requests to login pages. Handles both:
    - PDF served inline (captured via response interception)
    - PDF served as download (captured via expect_download)

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
    download_triggered = False

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
        logger.debug("Playwright navigating to PDF URL")
        timeout_ms = int(timeout * 1000)

        try:
            response = await page.goto(
                url, timeout=timeout_ms, wait_until="networkidle"
            )
        except Exception as nav_error:
            error_str = str(nav_error)
            if "Download is starting" in error_str:
                download_triggered = True
            else:
                raise

        # Check if we captured the PDF via response interception
        if content and validate_pdf_bytes(content):
            logger.debug(
                f"Playwright captured PDF via response: {len(content) / 1024:.1f} KB"
            )
            return content

        # If download was triggered, use expect_download to capture it
        if download_triggered:
            logger.debug("Download triggered, using expect_download to capture")
            content = await _capture_download_with_expect(page, url, timeout_ms)
            if content and validate_pdf_bytes(content):
                return content
            raise MarkerProcessingError(
                "Playwright download triggered but content is not a valid PDF"
            )

        # If not captured via response, check if we can get it from the response directly
        if response:
            content_type = response.headers.get("content-type", "")
            if "application/pdf" in content_type:
                content = await response.body()
                if validate_pdf_bytes(content):
                    logger.debug(
                        f"Playwright got PDF from response: {len(content) / 1024:.1f} KB"
                    )
                    return content

        raise MarkerProcessingError(
            "Playwright could not download PDF (got non-PDF content)"
        )

    except MarkerProcessingError:
        raise
    except Exception as e:
        raise MarkerProcessingError(f"Playwright PDF download failed: {e}") from e
    finally:
        await page.close()
        await context.close()


async def _capture_download_with_expect(page, url: str, timeout_ms: int) -> bytes:
    """Capture a PDF download using Playwright's expect_download.

    This is used when navigation triggers a download instead of loading a page.

    Args:
        page: Playwright page object
        url: URL that triggers the download
        timeout_ms: Timeout in milliseconds

    Returns:
        PDF content bytes
    """
    download_path: str | None = None

    try:
        async with page.expect_download(timeout=timeout_ms) as download_info:
            # Navigate again - this time we're ready for the download
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            except Exception:
                # Navigation will "fail" when download starts, that's expected
                pass

        # Wait for download to complete
        download = await download_info.value
        logger.debug(f"Download started: {download.suggested_filename}")

        # Save to temp file and read content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            download_path = tmp.name

        # save_as waits for download to complete
        await download.save_as(download_path)
        content = Path(download_path).read_bytes()
        logger.debug(
            f"Playwright captured download via expect_download: {len(content)} bytes"
        )
        return content

    finally:
        # Cleanup temp file
        if download_path:
            try:
                Path(download_path).unlink(missing_ok=True)
            except OSError:
                pass


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
        error_str = str(e)
        # Try Playwright for:
        # - Non-PDF responses (HTML login pages)
        # - HTTP 4xx errors (anti-bot blocking: 403, 418, 429, etc.)
        # - Timeout errors (slow sites may work with browser)
        if any(
            pattern in error_str
            for pattern in [
                "not a valid PDF",
                "HTTP error",
                "Timeout",
            ]
        ):
            logger.debug(f"httpx failed ({e}), trying Playwright fallback")
        else:
            raise

    # Fallback to Playwright (handles anti-bot redirects and blocks)
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
    max_retries: int = 3,
) -> str:
    """Submit a PDF conversion job to Marker.

    Args:
        file_path: Path relative to Marker input directory
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        max_retries: Max retries for transient network errors

    Returns:
        Job ID for polling
    """
    async with httpx.AsyncClient(base_url=MARKER_BASE_URL, timeout=60.0) as client:
        payload = {
            "file_path": file_path,
            "quality": quality,
            "markdown_only": False,
            "langs": langs or ["English"],
        }

        backoff_multipliers = (2, 5, 10)  # Longer backoffs for busy service
        for attempt in range(max_retries):
            try:
                response = await client.post("/convert", json=payload)
                response.raise_for_status()
                data = response.json()
                return data["job_id"]
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2.0 * backoff_multipliers[attempt]  # 4s, 10s, 20s
                    logger.warning(
                        f"Marker submit timeout (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise MarkerProcessingError(
                        f"Marker job submission failed after {max_retries} retries: {e}"
                    ) from e

        # Should never reach here, but satisfy type checker
        raise MarkerProcessingError("Marker job submission failed")


async def _poll_marker_job(
    job_id: str,
    max_wait: Optional[float] = None,
    max_retries: int = 3,
) -> str:
    """Poll Marker job until completion.

    Args:
        job_id: Job ID to poll
        max_wait: Maximum wait time in seconds (None = no limit)
        max_retries: Max retries for transient network errors per poll attempt

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If job fails or times out
    """
    start_time = asyncio.get_event_loop().time()

    # Use longer timeout for poll requests - Marker can be slow under load
    async with httpx.AsyncClient(base_url=MARKER_BASE_URL, timeout=60.0) as client:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if max_wait is not None and elapsed > max_wait:
                raise MarkerProcessingError(
                    f"Marker job {job_id} did not complete within {max_wait}s"
                )

            # Retry transient network errors with longer backoffs for busy service
            backoff_multipliers = (2, 5, 10)  # 30s, 75s, 150s
            for attempt in range(max_retries):
                try:
                    response = await client.get(f"/jobs/{job_id}")
                    response.raise_for_status()
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = MARKER_POLL_INTERVAL * backoff_multipliers[attempt]
                        logger.warning(
                            f"Marker poll timeout for job {job_id} (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise MarkerProcessingError(
                            f"Marker poll failed after {max_retries} retries: {e}"
                        ) from e

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
    timeout: Optional[float] = None,
) -> str:
    """Download PDF from URL and convert to markdown via Marker.

    Args:
        url: URL to PDF file
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        timeout: Maximum time for Marker processing (None = no limit)

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If download or processing fails
    """
    logger.debug("Processing PDF URL")

    # Download PDF (download timeout separate from Marker processing)
    content = await _download_pdf(url, timeout=60.0)
    logger.debug(f"Downloaded PDF: {len(content) / 1024:.1f} KB")

    # Process via Marker (no timeout - large PDFs can take a long time)
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
    timeout: Optional[float] = None,
    filename: Optional[str] = None,
) -> str:
    """Convert PDF bytes to markdown via Marker.

    Automatically chunks large PDFs (>100 pages by default) to prevent
    memory exhaustion. Chunks are processed sequentially and reassembled.

    Args:
        content: PDF content bytes
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        timeout: Maximum time for processing (None = no limit, uses Marker queue)
        filename: Optional filename for saved file

    Returns:
        Markdown content

    Raises:
        MarkerProcessingError: If processing fails
    """
    if not validate_pdf_bytes(content):
        raise MarkerProcessingError("Content is not a valid PDF")

    # Check file size limit
    if len(content) > MARKER_MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        limit_mb = MARKER_MAX_FILE_SIZE / (1024 * 1024)
        raise MarkerProcessingError(
            f"PDF too large ({size_mb:.1f}MB > {limit_mb:.0f}MB limit)"
        )

    # Check if PDF needs chunking
    if should_chunk_pdf(content, MARKER_CHUNK_PAGE_THRESHOLD):
        return await _process_chunked_pdf(
            content,
            quality=quality,
            langs=langs,
            timeout=timeout,
        )

    # Standard single-PDF processing
    return await _process_single_pdf(
        content,
        quality=quality,
        langs=langs,
        timeout=timeout,
        filename=filename,
    )


async def _process_single_pdf(
    content: bytes,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: Optional[float] = None,
    filename: Optional[str] = None,
) -> str:
    """Process a single PDF (internal helper).

    Args:
        content: PDF content bytes
        quality: Quality preset
        langs: Languages for OCR
        timeout: Maximum processing time
        filename: Optional filename

    Returns:
        Markdown content
    """
    # Save to Marker input directory
    file_path = _save_to_marker_input(content, filename)

    # Submit job
    job_id = await _submit_marker_job(file_path, quality=quality, langs=langs)
    logger.debug(f"Submitted Marker job: {job_id}")

    # Poll for completion
    markdown = await _poll_marker_job(job_id, max_wait=timeout)
    logger.debug(f"Marker conversion complete: {len(markdown)} chars")

    return markdown


async def _process_chunked_pdf(
    content: bytes,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Process a large PDF in chunks to prevent memory exhaustion.

    Splits the PDF into smaller chunks, processes each sequentially,
    then reassembles the markdown output.

    Args:
        content: PDF content bytes
        quality: Quality preset
        langs: Languages for OCR
        timeout: Maximum processing time per chunk

    Returns:
        Assembled markdown content
    """
    # Split PDF into chunks
    chunks = split_pdf_by_pages(content, MARKER_CHUNK_SIZE)
    logger.info(
        f"Processing large PDF in {len(chunks)} chunks of ~{MARKER_CHUNK_SIZE} pages"
    )

    markdown_chunks = []
    page_ranges = []

    for i, (chunk_bytes, page_range) in enumerate(chunks):
        chunk_num = i + 1
        logger.info(
            f"Processing chunk {chunk_num}/{len(chunks)} (pages {page_range[0]}-{page_range[1]})"
        )

        try:
            # Process this chunk
            chunk_filename = f"chunk_{chunk_num}_of_{len(chunks)}.pdf"
            markdown = await _process_single_pdf(
                chunk_bytes,
                quality=quality,
                langs=langs,
                timeout=timeout,
                filename=chunk_filename,
            )
            markdown_chunks.append(markdown)
            page_ranges.append(page_range)

            logger.debug(
                f"Chunk {chunk_num}/{len(chunks)} complete: {len(markdown)} chars"
            )

        except Exception as e:
            logger.error(f"Chunk {chunk_num}/{len(chunks)} failed: {e}")
            raise MarkerProcessingError(
                f"Failed processing chunk {chunk_num} (pages {page_range[0]}-{page_range[1]}): {e}"
            ) from e

        finally:
            # Aggressive memory cleanup between chunks
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # torch not available in this context

    # Reassemble chunks
    assembled = assemble_markdown_chunks(markdown_chunks, page_ranges)
    logger.info(
        f"Assembled {len(chunks)} chunks into {len(assembled)} chars of markdown"
    )

    return assembled


async def process_pdf_file(
    file_path: str,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Convert PDF file to markdown via Marker.

    Args:
        file_path: Path to PDF file
        quality: Quality preset (fast, balanced, quality)
        langs: Languages for OCR
        timeout: Maximum time for processing (None = no limit, uses Marker queue)

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


async def download_pdf_by_md5(
    md5: str,
    output_path: Path,
    identifier: Optional[str] = None,
    timeout: float = 120.0,
) -> Optional[str]:
    """Download PDF via MD5 hash without processing. Returns local file path.

    Uses the retrieve-academic service (VPN-enabled) to download PDFs by MD5 hash.
    The PDF is saved to the output_path without any Marker conversion.

    Args:
        md5: MD5 hash of the document to download
        output_path: Path where the PDF should be saved
        identifier: Identifier for logging (e.g., book title)
        timeout: Download timeout in seconds

    Returns:
        Local file path (str) if successful, None if failed.
    """
    from core.stores import RetrieveAcademicClient

    logger.debug(f"Downloading PDF by MD5: {md5[:12]}...")

    try:
        async with RetrieveAcademicClient() as client:
            if not await client.health_check():
                logger.warning("retrieve-academic service unavailable")
                return None

            # Ensure parent directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            await client.download_by_md5(
                md5=md5,
                local_path=str(output_path),
                identifier=identifier or md5,
                timeout=timeout,
            )

            if output_path.exists():
                logger.debug(
                    f"Downloaded PDF: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)"
                )
                return str(output_path)
            else:
                logger.warning(f"Download completed but file not found: {output_path}")
                return None

    except Exception as e:
        logger.warning(f"PDF download failed for md5={md5[:12]}...: {e}")
        return None


async def process_pdf_by_md5(
    md5: str,
    identifier: Optional[str] = None,
    quality: str = "fast",
    langs: Optional[list[str]] = None,
    download_timeout: float = 120.0,
    marker_timeout: Optional[float] = None,
) -> Optional[str]:
    """Download PDF via MD5 hash (Anna's Archive pattern) and convert via Marker.

    Uses the retrieve-academic service (VPN-enabled) to download PDFs by MD5 hash,
    then converts to markdown using Marker.

    Args:
        md5: MD5 hash of the document to download
        identifier: Identifier for logging (e.g., book title)
        quality: Quality preset (fast, balanced, quality). Defaults to "fast".
        langs: Languages for OCR
        download_timeout: Download timeout in seconds (for retrieve-academic)
        marker_timeout: Marker processing timeout (None = no limit, uses Marker queue)

    Returns:
        Markdown content or None if failed.
    """
    from core.stores import RetrieveAcademicClient

    logger.debug(f"Processing PDF by MD5: {md5[:12]}...")

    try:
        async with RetrieveAcademicClient() as client:
            if not await client.health_check():
                logger.warning("retrieve-academic service unavailable")
                return None

            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                await client.download_by_md5(
                    md5=md5,
                    local_path=tmp_path,
                    identifier=identifier or md5,
                    timeout=download_timeout,
                )

                # Convert via Marker (no timeout - uses Marker's native queue)
                return await process_pdf_file(
                    tmp_path,
                    quality=quality,
                    langs=langs,
                    timeout=marker_timeout,
                )
            finally:
                # Cleanup temp file
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    except Exception as e:
        logger.warning(f"PDF processing failed for md5={md5[:12]}...: {e}")
        return None
