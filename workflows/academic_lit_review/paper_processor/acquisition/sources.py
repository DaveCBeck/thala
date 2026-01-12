"""Source-specific paper acquisition logic."""

import logging
from pathlib import Path
from typing import Optional

import httpx

from langchain_tools.firecrawl import scrape_url
from workflows.shared import download_url, ContentTypeError
from workflows.academic_lit_review.paper_processor.classification import classify_scraped_content

from .http_client import _is_pdf_url, _download_pdf_from_url
from .types import OA_DOWNLOAD_TIMEOUT

logger = logging.getLogger(__name__)


async def try_oa_download(
    oa_url: str,
    local_path: Path,
    doi: str,
) -> tuple[Optional[str], bool]:
    """Try to download paper from Open Access URL.

    Handles both PDF URLs (direct download) and HTML URLs (firecrawl scrape).
    For HTML pages, uses LLM classification to determine:
    - full_text: Return markdown content directly
    - abstract_with_pdf: Extract PDF URL and download the PDF
    - paywall: Return None to trigger retrieve-academic fallback

    Args:
        oa_url: Open Access URL from OpenAlex
        local_path: Path to save PDF (if PDF URL or extracted from abstract page)
        doi: DOI for logging

    Returns:
        Tuple of (source, is_markdown):
        - For PDF: (local_path_str, False) on success
        - For HTML full text: (markdown_content, True) on success
        - For abstract+PDF: downloads PDF, returns (local_path_str, False)
        - (None, False) on failure or paywall
    """
    try:
        if _is_pdf_url(oa_url):
            logger.info(f"[OA] Downloading PDF for {doi}: {oa_url}")
            try:
                content = await download_url(
                    oa_url,
                    timeout=OA_DOWNLOAD_TIMEOUT,
                    expected_content_type="pdf",
                    validate_pdf=True,
                )

                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(content)

                logger.info(f"[OA] Downloaded PDF for {doi}: {len(content) / 1024:.1f} KB")
                return str(local_path), False

            except ContentTypeError:
                logger.warning(f"[OA] URL returned non-PDF content for {doi}")
                return None, False

        else:
            logger.info(f"[OA] Scraping HTML page for {doi}: {oa_url}")
            response = await scrape_url.ainvoke({"url": oa_url, "include_links": True})
            markdown = response.get("markdown", "")
            links = response.get("links", [])

            if not markdown or len(markdown) < 500:
                logger.warning(f"[OA] Scraped content too short for {doi}: {len(markdown)} chars")
                return None, False

            classification = await classify_scraped_content(
                doi=doi,
                url=oa_url,
                markdown=markdown,
                links=links,
            )
            logger.info(
                f"[OA] Classification for {doi}: {classification.classification} "
                f"(confidence={classification.confidence:.2f})"
            )

            if classification.classification == "paywall":
                logger.info(
                    f"[OA] Paywall detected for {doi}, falling back to retrieve-academic"
                )
                return None, False

            if classification.classification == "abstract_with_pdf":
                pdf_url = classification.pdf_url
                if pdf_url and len(pdf_url) < 2000 and pdf_url.startswith(("http://", "https://")):
                    logger.info(
                        f"[OA] Abstract page with PDF link for {doi}: {pdf_url}"
                    )
                    return await _download_pdf_from_url(pdf_url, local_path, doi)
                else:
                    logger.warning(
                        f"[OA] Abstract page but no valid PDF URL for {doi} "
                        f"(got: {pdf_url[:100] if pdf_url else None}...), "
                        f"falling back to retrieve-academic"
                    )
                    return None, False

            logger.info(f"[OA] Full text scraped for {doi}: {len(markdown)} chars")
            return markdown, True

    except httpx.HTTPStatusError as e:
        logger.warning(f"[OA] HTTP error for {doi}: {e.response.status_code}")
        return None, False
    except Exception as e:
        logger.warning(f"[OA] Failed to download from OA URL for {doi}: {type(e).__name__}: {e}")
        return None, False
