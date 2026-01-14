"""Unified URL content retrieval system.

Single entry point for fetching content from any URL type:
- Regular web URLs
- DOIs (bare or URL form)
- PDF URLs
- Academic publisher pages

Automatically detects content type and applies appropriate processing
with intelligent fallback chains.

Example usage:
    from core.scraping import get_url

    # Regular URL
    result = await get_url("https://example.com")

    # DOI (bare or URL)
    result = await get_url("10.1038/nature12373")
    result = await get_url("https://doi.org/10.1038/nature12373")

    # PDF URL
    result = await get_url("https://arxiv.org/pdf/2301.00001.pdf")

    # With options
    result = await get_url(
        "10.1234/example",
        GetUrlOptions(
            pdf_quality="quality",
            allow_retrieve_academic=True,
        ),
    )
"""

import logging
from typing import Optional

from .classification import classify_content, ClassificationResult
from .doi import detect_doi, extract_doi_from_content, get_oa_url_for_doi, search_doi_by_title
from .fallback import try_retrieve_academic
from .pdf import is_pdf_url, process_pdf_url
from .service import get_scraper_service
from .types import (
    ContentClassification,
    ContentSource,
    DoiInfo,
    GetUrlOptions,
    GetUrlResult,
)

logger = logging.getLogger(__name__)


async def get_url(
    url: str,
    options: Optional[GetUrlOptions] = None,
) -> GetUrlResult:
    """Unified URL content retrieval.

    Handles:
    1. DOI detection & resolution via OpenAlex
    2. PDF detection & processing via Marker
    3. Automatic academic content detection
    4. Fallback chain for access issues

    Args:
        url: URL, DOI, or doi.org URL
        options: Processing options

    Returns:
        GetUrlResult with markdown content and metadata

    Raises:
        Exception: If all retrieval methods fail
    """
    opts = options or GetUrlOptions()
    fallback_chain: list[str] = []
    doi_info: Optional[DoiInfo] = None
    resolved_url = url

    # Step 1: DOI Detection
    doi_info = detect_doi(url)
    if doi_info:
        logger.debug(f"DOI detected: {doi_info.doi} (source: {doi_info.source})")
        fallback_chain.append("doi_detected")

        # Try to get OA URL from OpenAlex
        oa_url = await get_oa_url_for_doi(doi_info.doi)
        if oa_url:
            resolved_url = oa_url
            logger.debug(f"Resolved DOI to OA URL: {oa_url}")
            fallback_chain.append("openalex_oa_url")
        else:
            # No OA URL, use DOI URL as fallback
            resolved_url = doi_info.doi_url
            logger.debug(f"No OA URL found, using DOI URL")
            fallback_chain.append("doi_url_fallback")

    # Step 2: PDF Detection
    is_pdf = is_pdf_url(resolved_url)
    if is_pdf:
        fallback_chain.append("pdf_direct")
        logger.debug(f"PDF URL detected")

        result = await _handle_pdf_url(resolved_url, doi_info, opts, fallback_chain)
        if result:
            return result
        # PDF download failed - skip web scraping (won't work for PDFs) and go to retrieve-academic
        logger.warning(f"PDF download failed, skipping web scraping for PDF URL")
        # Jump directly to retrieve-academic fallback (Step 5)
        if opts.allow_retrieve_academic and doi_info:
            fallback_chain.append("retrieve_academic")
            result = await try_retrieve_academic(doi_info.doi, opts, fallback_chain)
            if result:
                return result
        # All methods failed for PDF
        error_msg = f"Failed to retrieve PDF from {resolved_url}"
        if fallback_chain:
            error_msg += f" (attempted: {' -> '.join(fallback_chain)})"
        raise Exception(error_msg)

    # Step 3: Web Scraping (only for non-PDF URLs)
    fallback_chain.append("scraper_service")
    scraper = get_scraper_service()

    scrape_result = None
    try:
        scrape_result = await scraper.scrape(resolved_url, include_links=opts.include_links)
        logger.debug(f"Scraped {len(scrape_result.markdown)} chars via {scrape_result.provider}")
    except Exception as e:
        logger.warning(f"Scraping failed: {type(e).__name__}: {e}")

    # Step 4: Content Classification (if enabled and scrape succeeded)
    if scrape_result and opts.detect_academic:
        classification = await classify_content(
            url=resolved_url,
            markdown=scrape_result.markdown,
            links=scrape_result.links,
            doi=doi_info.doi if doi_info else None,
        )
        fallback_chain.append(f"classified:{classification.classification}")

        # Extract DOI from content if not already known
        if not doi_info:
            content_doi = extract_doi_from_content(scrape_result.markdown)
            if content_doi:
                doi_info = detect_doi(content_doi)
                if doi_info:
                    logger.debug(f"DOI extracted from content: {doi_info.doi}")
                    fallback_chain.append("doi_from_content")

        # Handle based on classification
        if classification.classification == "full_text":
            return GetUrlResult(
                url=url,
                resolved_url=resolved_url,
                content=scrape_result.markdown,
                source=ContentSource.SCRAPED,
                provider=scrape_result.provider,
                doi=doi_info.doi if doi_info else None,
                classification=ContentClassification.FULL_TEXT,
                links=scrape_result.links,
                fallback_chain=fallback_chain,
            )

        elif classification.classification == "abstract_with_pdf":
            if classification.pdf_url:
                fallback_chain.append("pdf_extracted")
                logger.debug(f"Extracted PDF URL from abstract page")

                pdf_result = await _handle_pdf_url(
                    classification.pdf_url, doi_info, opts, fallback_chain
                )
                if pdf_result:
                    pdf_result.classification = ContentClassification.ABSTRACT_WITH_PDF
                    return pdf_result
                # Fall through to retrieve-academic
                logger.warning(f"Extracted PDF download failed")
            else:
                logger.warning(f"Abstract page but no valid PDF URL extracted")
            # Fall through to retrieve-academic

        elif classification.classification == "paywall":
            logger.debug(f"Paywall detected")
            fallback_chain.append("paywall_detected")

            # If no DOI known, try to find it via title search
            if not doi_info and classification.title:
                logger.debug(f"Searching OpenAlex for DOI by title: {classification.title[:50]}...")
                found_doi = await search_doi_by_title(
                    classification.title,
                    classification.authors,
                )
                if found_doi:
                    doi_info = DoiInfo(
                        doi=found_doi,
                        doi_url=f"https://doi.org/{found_doi}",
                        source="title_search",
                    )
                    fallback_chain.append("doi_from_title_search")
                    logger.debug(f"Found DOI via title search: {found_doi}")
            # Fall through to retrieve-academic

        elif classification.classification == "non_academic":
            # Non-academic content - return as-is
            return GetUrlResult(
                url=url,
                resolved_url=resolved_url,
                content=scrape_result.markdown,
                source=ContentSource.SCRAPED,
                provider=scrape_result.provider,
                doi=doi_info.doi if doi_info else None,
                classification=ContentClassification.NON_ACADEMIC,
                links=scrape_result.links,
                fallback_chain=fallback_chain,
            )

    elif scrape_result:
        # No classification requested, return scraped content
        return GetUrlResult(
            url=url,
            resolved_url=resolved_url,
            content=scrape_result.markdown,
            source=ContentSource.SCRAPED,
            provider=scrape_result.provider,
            doi=doi_info.doi if doi_info else None,
            links=scrape_result.links,
            fallback_chain=fallback_chain,
        )

    # Step 5: Retrieve-Academic Fallback
    if opts.allow_retrieve_academic and doi_info:
        fallback_chain.append("retrieve_academic")
        result = await try_retrieve_academic(doi_info.doi, opts, fallback_chain)
        if result:
            return result

    # All methods failed
    error_msg = f"Failed to retrieve content from {url}"
    if fallback_chain:
        error_msg += f" (attempted: {' -> '.join(fallback_chain)})"
    raise Exception(error_msg)


async def _handle_pdf_url(
    url: str,
    doi_info: Optional[DoiInfo],
    opts: GetUrlOptions,
    fallback_chain: list[str],
) -> Optional[GetUrlResult]:
    """Download and process PDF URL.

    Args:
        url: PDF URL to download
        doi_info: DOI info if known
        opts: Processing options
        fallback_chain: List of attempted sources

    Returns:
        GetUrlResult if successful, None otherwise
    """
    try:
        markdown = await process_pdf_url(
            url,
            quality=opts.pdf_quality,
            langs=opts.pdf_langs,
            timeout=opts.pdf_timeout,
        )
        return GetUrlResult(
            url=url,
            resolved_url=url,
            content=markdown,
            source=ContentSource.PDF_DIRECT,
            provider="marker",
            doi=doi_info.doi if doi_info else None,
            fallback_chain=fallback_chain,
        )
    except Exception as e:
        logger.warning(f"PDF processing failed: {type(e).__name__}: {e}")
        return None
