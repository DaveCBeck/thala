"""Retrieve-academic fallback for paywalled content.

Uses the retrieve-academic service to get full-text academic documents
when open access URLs are not available.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from core.stores import RetrieveAcademicClient

from ..pdf.processor import process_pdf_file
from ..types import ContentSource, GetUrlOptions, GetUrlResult

logger = logging.getLogger(__name__)


async def try_retrieve_academic(
    doi: str,
    options: GetUrlOptions,
    fallback_chain: list[str],
) -> Optional[GetUrlResult]:
    """Try retrieve-academic service for paywalled content.

    Args:
        doi: DOI to retrieve
        options: Processing options
        fallback_chain: List of attempted sources (for debugging)

    Returns:
        GetUrlResult if successful, None otherwise
    """
    logger.info(f"Attempting retrieve-academic fallback for DOI: {doi}")

    try:
        async with RetrieveAcademicClient() as client:
            # Check if service is available
            if not await client.health_check():
                logger.warning("retrieve-academic service unavailable (VPN not connected)")
                return None

            # Submit retrieval job
            job = await client.retrieve(
                doi=doi,
                timeout_seconds=int(options.retrieve_academic_timeout),
            )
            logger.info(f"Submitted retrieve-academic job: {job.job_id}")

            # Wait for completion
            result = await client.wait_for_completion(
                job.job_id,
                timeout=options.retrieve_academic_timeout,
            )

            if result.status != "completed":
                logger.warning(
                    f"retrieve-academic job failed: {result.error_code} - {result.error}"
                )
                return None

            # Download file to temp location
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = Path(temp_dir) / f"{doi.replace('/', '_')}.pdf"
                await client.download_file(job.job_id, str(local_path))

                logger.info(f"Downloaded via retrieve-academic: {local_path}")

                # Process PDF via Marker
                markdown = await process_pdf_file(
                    str(local_path),
                    quality=options.pdf_quality,
                    langs=options.pdf_langs,
                    timeout=options.pdf_timeout,
                )

                return GetUrlResult(
                    url=f"doi:{doi}",
                    resolved_url=f"https://doi.org/{doi}",
                    content=markdown,
                    source=ContentSource.RETRIEVE_ACADEMIC,
                    provider="retrieve-academic",
                    doi=doi,
                    fallback_chain=fallback_chain,
                )

    except Exception as e:
        logger.error(f"retrieve-academic fallback failed for {doi}: {type(e).__name__}: {e}")
        return None
