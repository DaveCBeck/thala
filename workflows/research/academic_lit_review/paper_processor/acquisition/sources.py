"""Source-specific paper acquisition logic."""

import logging
from pathlib import Path
from typing import Optional

from core.scraping import get_url, GetUrlOptions, ContentClassification
from core.scraping.types import ContentSource

logger = logging.getLogger(__name__)


async def try_oa_download(
    oa_url: str,
    local_path: Path,
    doi: str,
) -> tuple[Optional[str], bool, Optional[ContentSource], Optional[str]]:
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
        Tuple of (content, is_markdown, content_source, provider):
        - (markdown_content, True, ContentSource, provider_str) on success
        - (None, False, None, None) on failure or paywall
    """
    try:
        logger.debug(f"[OA] Fetching content for {doi}: {oa_url}")

        result = await get_url(
            oa_url,
            GetUrlOptions(
                detect_academic=True,
                allow_retrieve_academic=False,  # Pipeline handles this separately
            ),
        )

        # Check for paywall
        if result.classification == ContentClassification.PAYWALL:
            logger.debug(f"[OA] Paywall detected for {doi}, falling back to retrieve-academic")
            return None, False, None, None

        logger.debug(
            f"[OA] Got content for {doi}: {len(result.content)} chars "
            f"(source={result.source.value}, provider={result.provider}, "
            f"classification={result.classification})"
        )

        return result.content, True, result.source, result.provider

    except Exception as e:
        logger.debug(f"[OA] Failed to download from OA URL for {doi}: {type(e).__name__}: {e}")
        return None, False, None, None
