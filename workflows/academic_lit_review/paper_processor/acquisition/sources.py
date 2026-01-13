"""Source-specific paper acquisition logic."""

import logging
from pathlib import Path
from typing import Optional

from core.scraping import get_url, GetUrlOptions, ContentClassification, ContentSource

logger = logging.getLogger(__name__)


async def try_oa_download(
    oa_url: str,
    local_path: Path,
    doi: str,
) -> tuple[Optional[str], bool]:
    """Try to download paper from Open Access URL.

    Uses unified get_url() which handles:
    - PDF detection and download via Marker
    - HTML scraping via Firecrawl cascade
    - Content classification (paywall, full_text, abstract_with_pdf)
    - PDF extraction from abstract pages

    Args:
        oa_url: Open Access URL from OpenAlex
        local_path: Path to save PDF (unused - get_url returns markdown directly)
        doi: DOI for logging

    Returns:
        Tuple of (content, is_markdown):
        - (markdown_content, True) on success
        - (None, False) on failure or paywall
    """
    try:
        logger.info(f"[OA] Fetching content for {doi}: {oa_url}")

        result = await get_url(
            oa_url,
            GetUrlOptions(
                detect_academic=True,
                allow_retrieve_academic=False,  # Pipeline handles this separately
            ),
        )

        # Check for paywall
        if result.classification == ContentClassification.PAYWALL:
            logger.info(f"[OA] Paywall detected for {doi}, falling back to retrieve-academic")
            return None, False

        # Log source info
        source_info = f"source={result.source.value}"
        if result.provider:
            source_info += f", provider={result.provider}"

        logger.info(
            f"[OA] Got content for {doi}: {len(result.content)} chars "
            f"({source_info}, classification={result.classification})"
        )

        return result.content, True

    except Exception as e:
        logger.warning(f"[OA] Failed to download from OA URL for {doi}: {type(e).__name__}: {e}")
        return None, False
