"""HTTP client utilities for downloading papers."""

import logging
from pathlib import Path
from typing import Optional

import httpx

from workflows.shared import download_url, ContentTypeError

from .types import OA_DOWNLOAD_TIMEOUT

logger = logging.getLogger(__name__)


def _is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    clean_url = url.lower().split("?")[0].split("#")[0].rstrip("/")
    return clean_url.endswith(".pdf")


async def _download_pdf_from_url(
    pdf_url: str,
    local_path: Path,
    doi: str,
) -> tuple[Optional[str], bool]:
    """Download PDF from an extracted URL.

    Args:
        pdf_url: URL to the PDF file
        local_path: Where to save the PDF
        doi: DOI for logging

    Returns:
        (local_path_str, False) on success, (None, False) on failure
    """
    try:
        content = await download_url(
            pdf_url,
            timeout=OA_DOWNLOAD_TIMEOUT,
            expected_content_type="pdf",
            validate_pdf=True,
        )

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)

        logger.info(
            f"[OA] Downloaded PDF from abstract page for {doi}: "
            f"{len(content) / 1024:.1f} KB"
        )
        return str(local_path), False

    except ContentTypeError as e:
        logger.warning(f"[OA] Extracted PDF URL returned non-PDF for {doi}: {e}")
        return None, False
    except httpx.HTTPStatusError as e:
        logger.warning(f"[OA] HTTP error downloading PDF for {doi}: {e.response.status_code}")
        return None, False
    except Exception as e:
        logger.warning(
            f"[OA] Failed to download PDF from {pdf_url} for {doi}: "
            f"{type(e).__name__}: {e}"
        )
        return None, False
