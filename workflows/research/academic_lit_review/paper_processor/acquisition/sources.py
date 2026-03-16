"""Source-specific paper acquisition logic."""

import logging
from pathlib import Path
from typing import Optional

import httpx

from core.scraping import get_url, GetUrlOptions, ContentClassification
from core.scraping.types import ContentSource

logger = logging.getLogger(__name__)

DOI_RESOLVE_URL = "https://doi.org/"


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

        # Reject suspiciously short content — error pages, service-unavailable
        # notices, and stub pages are never valid full-text papers.
        MIN_FULL_TEXT_CHARS = 500
        if len(result.content) < MIN_FULL_TEXT_CHARS:
            logger.debug(
                f"[OA] Content too short for {doi} ({len(result.content)} chars < {MIN_FULL_TEXT_CHARS}), "
                f"likely an error page — falling back"
            )
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


async def try_pmc_download(
    pmcid: str,
    doi: str,
    local_path: Path,
) -> tuple[Optional[str], bool, Optional[ContentSource], Optional[str]]:
    """Try to download paper from PubMed Central.

    PMC provides free full-text for biomedical papers. No API key or rate
    limit concerns.

    Returns same tuple format as try_oa_download.
    """
    pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    try:
        logger.debug(f"[PMC] Fetching content for {doi}: {pmc_url}")

        result = await get_url(
            pmc_url,
            GetUrlOptions(
                detect_academic=True,
                allow_retrieve_academic=False,
            ),
        )

        if result.classification == ContentClassification.PAYWALL:
            logger.debug(f"[PMC] Unexpected paywall for {doi}")
            return None, False, None, None

        MIN_FULL_TEXT_CHARS = 500
        if len(result.content) < MIN_FULL_TEXT_CHARS:
            logger.debug(f"[PMC] Content too short for {doi} ({len(result.content)} chars), likely error page")
            return None, False, None, None

        logger.debug(
            f"[PMC] Got content for {doi}: {len(result.content)} chars "
            f"(source={result.source.value}, provider={result.provider})"
        )

        return result.content, True, result.source, result.provider

    except Exception as e:
        logger.debug(f"[PMC] Failed to download for {doi}: {type(e).__name__}: {e}")
        return None, False, None, None


async def resolve_doi_url(doi: str) -> Optional[str]:
    """Resolve a DOI to its publisher URL by following the doi.org redirect.

    Returns the final URL after redirects, or None on failure.
    """
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"Accept": "text/html"},
        ) as client:
            response = await client.head(f"{DOI_RESOLVE_URL}{doi}")
            publisher_url = str(response.url)
            if publisher_url and publisher_url != f"{DOI_RESOLVE_URL}{doi}":
                return publisher_url
    except Exception as e:
        logger.debug(f"[DOI] Failed to resolve {doi}: {type(e).__name__}: {e}")
    return None


async def try_doi_download(
    doi: str,
    local_path: Path,
    failed_hosts: set[str] | None = None,
) -> tuple[Optional[str], bool, Optional[ContentSource], Optional[str]]:
    """Try to download paper by resolving DOI to publisher URL.

    Resolves doi.org/{doi} to the publisher page, then uses get_url()
    to scrape full-text HTML. Returns None on paywall or failure.

    If failed_hosts is provided, skips the fetch when the resolved URL
    lands on a host that already failed (avoids wasting a Firecrawl scrape).

    Returns same tuple format as try_oa_download.
    """
    publisher_url = await resolve_doi_url(doi)
    if not publisher_url:
        return None, False, None, None

    # Skip if the resolved host already failed during OA attempts
    if failed_hosts:
        from urllib.parse import urlparse
        resolved_host = urlparse(publisher_url).netloc
        if resolved_host in failed_hosts:
            logger.debug(f"[DOI] Skipping DOI resolution — same host already failed: {resolved_host} ({doi})")
            return None, False, None, None

    logger.debug(f"[DOI] Resolved {doi} -> {publisher_url}")

    try:
        result = await get_url(
            publisher_url,
            GetUrlOptions(
                detect_academic=True,
                allow_retrieve_academic=False,
            ),
        )

        if result.classification == ContentClassification.PAYWALL:
            logger.debug(f"[DOI] Paywall at publisher URL for {doi}")
            return None, False, None, None

        MIN_FULL_TEXT_CHARS = 500
        if len(result.content) < MIN_FULL_TEXT_CHARS:
            logger.debug(f"[DOI] Content too short for {doi} ({len(result.content)} chars), likely error page")
            return None, False, None, None

        logger.debug(
            f"[DOI] Got content for {doi}: {len(result.content)} chars "
            f"(source={result.source.value}, provider={result.provider}, "
            f"classification={result.classification})"
        )

        return result.content, True, result.source, result.provider

    except Exception as e:
        logger.debug(f"[DOI] Failed to fetch publisher URL for {doi}: {type(e).__name__}: {e}")
        return None, False, None, None
