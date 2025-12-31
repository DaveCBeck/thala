"""PDF processing via Marker service."""

import logging
import os
import uuid

import httpx

from workflows.shared.marker_client import MarkerClient

logger = logging.getLogger(__name__)

MARKER_INPUT_DIR = os.getenv(
    "MARKER_INPUT_DIR",
    "/home/dave/thala/services/marker/data/input"
)


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    return url.lower().rstrip('/').endswith('.pdf')


async def fetch_pdf_via_marker(url: str) -> str | None:
    """Download PDF and convert via Marker service.

    Returns markdown content or None if failed.
    """
    filename = f"{uuid.uuid4().hex}.pdf"
    input_path = os.path.join(MARKER_INPUT_DIR, filename)

    try:
        # Download PDF
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        # Write to Marker input directory
        with open(input_path, "wb") as f:
            f.write(response.content)

        # Convert via Marker
        async with MarkerClient() as marker:
            result = await marker.convert(
                file_path=filename,
                quality="fast",  # Fast preset for research scraping
            )
            return result.markdown

    except Exception as e:
        logger.warning(f"Marker PDF processing failed for {url}: {e}")
        return None

    finally:
        # Cleanup temp file
        try:
            os.remove(input_path)
        except OSError:
            pass
