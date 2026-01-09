"""PDF processing via Marker service."""

import logging
import os
import uuid

from core.stores.retrieve_academic import RetrieveAcademicClient
from workflows.shared import download_url
from workflows.shared.marker_client import MarkerClient

logger = logging.getLogger(__name__)

MARKER_INPUT_DIR = os.getenv(
    "MARKER_INPUT_DIR",
    "/home/dave/thala/services/marker/data/input"
)


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    return url.lower().rstrip('/').endswith('.pdf')


async def fetch_pdf_via_marker(
    md5: str,
    identifier: str | None = None,
) -> str | None:
    """Download PDF via VPN service and convert via Marker.

    Downloads the PDF through the retrieve-academic service (which uses VPN)
    and then converts it to markdown using Marker.

    Args:
        md5: MD5 hash of the document to download
        identifier: Identifier for file naming (e.g., book title)

    Returns:
        Markdown content or None if failed.
    """
    filename = f"{uuid.uuid4().hex}.pdf"
    input_path = os.path.join(MARKER_INPUT_DIR, filename)

    try:
        # Download PDF via VPN-enabled service
        async with RetrieveAcademicClient() as client:
            if not await client.health_check():
                logger.warning("Retrieve-academic service unavailable")
                return None

            await client.download_by_md5(
                md5=md5,
                local_path=input_path,
                identifier=identifier or md5,
                timeout=120.0,
            )

        # Convert via Marker
        async with MarkerClient() as marker:
            result = await marker.convert(
                file_path=filename,
                quality="fast",  # Fast preset for research scraping
            )
            return result.markdown

    except Exception as e:
        logger.warning(f"Marker PDF processing failed for md5={md5[:12]}...: {e}")
        return None

    finally:
        # Cleanup temp file
        try:
            os.remove(input_path)
        except OSError:
            pass


async def fetch_pdf_from_url(url: str) -> str | None:
    """Download PDF from URL and convert via Marker.

    This is for general web PDFs (not from Anna's Archive).
    Downloads directly without VPN.

    Args:
        url: URL of the PDF to download

    Returns:
        Markdown content or None if failed.
    """
    filename = f"{uuid.uuid4().hex}.pdf"
    input_path = os.path.join(MARKER_INPUT_DIR, filename)

    try:
        # Download PDF directly
        content = await download_url(url, timeout=60.0)

        # Write to Marker input directory
        with open(input_path, "wb") as f:
            f.write(content)

        # Convert via Marker
        async with MarkerClient() as marker:
            result = await marker.convert(
                file_path=filename,
                quality="fast",
            )
            return result.markdown

    except Exception as e:
        logger.warning(f"PDF processing failed for {url}: {e}")
        return None

    finally:
        # Cleanup temp file
        try:
            os.remove(input_path)
        except OSError:
            pass
