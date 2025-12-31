"""Paper acquisition and processing pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from core.stores.retrieve_academic import RetrieveAcademicClient
from workflows.research.subgraphs.academic_lit_review.state import PaperMetadata

from .cache import check_document_exists_by_doi
from .document_processing import process_single_document
from .types import (
    ACQUISITION_DELAY,
    ACQUISITION_TIMEOUT,
    MAX_PAPER_PIPELINE_CONCURRENT,
)

logger = logging.getLogger(__name__)


async def acquire_full_text(
    paper: PaperMetadata,
    client: RetrieveAcademicClient,
    output_dir: Path,
) -> Optional[str]:
    """Acquire full text for a single paper.

    Args:
        paper: Paper metadata
        client: Retrieve academic client
        output_dir: Directory to save downloaded files

    Returns:
        Local file path or None on failure
    """
    doi = paper.get("doi")
    title = paper.get("title", "Unknown")

    oa_url = paper.get("oa_url")
    if oa_url:
        logger.debug(f"Paper {doi} has OA URL: {oa_url}")

    authors = []
    for author in paper.get("authors", [])[:5]:
        name = author.get("name")
        if name:
            authors.append(name)

    safe_doi = doi.replace("/", "_").replace(":", "_")
    local_path = output_dir / f"{safe_doi}.pdf"

    try:
        saved_path, result = await client.retrieve_and_download(
            doi=doi,
            local_path=str(local_path),
            title=title,
            authors=authors,
            timeout=ACQUISITION_TIMEOUT,
        )
        logger.info(f"Acquired full text for {doi}: {saved_path}")
        return saved_path

    except Exception as e:
        logger.warning(f"Failed to acquire full text for {doi}: {e}")
        return None


async def acquire_and_process_single_paper(
    paper: PaperMetadata,
    client: RetrieveAcademicClient,
    output_dir: Path,
    paper_index: int,
    total_papers: int,
) -> dict[str, Any]:
    """Acquire full text and process a single paper as one unit.

    This combines acquisition and processing to naturally rate-limit
    retrieval requests (processing takes time, giving the source a break).

    Args:
        paper: Paper metadata
        client: Retrieve academic client
        output_dir: Directory to save downloaded files
        paper_index: Current paper number (for logging)
        total_papers: Total papers being processed

    Returns:
        Dict with acquisition and processing results
    """
    doi = paper.get("doi")
    title = paper.get("title", "Unknown")[:50]

    logger.info(f"[{paper_index}/{total_papers}] Processing: {title}...")

    result = {
        "doi": doi,
        "acquired": False,
        "local_path": None,
        "processing_result": None,
        "processing_success": False,
        "from_cache": False,
    }

    existing = await check_document_exists_by_doi(doi)
    if existing:
        logger.info(f"[{paper_index}/{total_papers}] Cache hit for {doi}, skipping download")
        result["acquired"] = True
        result["processing_success"] = True
        result["from_cache"] = True
        result["processing_result"] = {
            "doi": doi,
            "success": True,
            "es_record_id": existing["es_record_id"],
            "zotero_key": existing["zotero_key"],
            "short_summary": existing["short_summary"],
            "errors": [],
        }
        return result

    local_path = await acquire_full_text(paper, client, output_dir)

    if not local_path:
        logger.warning(f"[{paper_index}/{total_papers}] Acquisition failed for {doi}")
        return result

    result["acquired"] = True
    result["local_path"] = local_path

    processing_result = await process_single_document(doi, local_path, paper)
    result["processing_result"] = processing_result
    result["processing_success"] = processing_result.get("success", False)

    if result["processing_success"]:
        logger.info(f"[{paper_index}/{total_papers}] Completed: {title}")
    else:
        logger.warning(f"[{paper_index}/{total_papers}] Processing failed for {doi}")

    return result


async def run_paper_pipeline(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_PAPER_PIPELINE_CONCURRENT,
) -> tuple[dict[str, str], dict[str, dict], list[str], list[str]]:
    """Run acquire→process pipeline for all papers with controlled concurrency.

    Uses a unified pipeline where each paper goes through acquisition and
    processing as one unit. This naturally rate-limits retrieval because
    document processing takes significant time.

    Args:
        papers: Papers to process
        max_concurrent: Maximum concurrent paper pipelines (default: 2)

    Returns:
        Tuple of (acquired, processing_results, acquisition_failed, processing_failed)
    """
    async with RetrieveAcademicClient() as client:
        if not await client.health_check():
            logger.warning("Retrieve-academic service unavailable, skipping full-text acquisition")
            return {}, {}, [p.get("doi") for p in papers], []

        output_dir = Path("/tmp/thala_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        acquired = {}
        processing_results = {}
        acquisition_failed = []
        processing_failed = []
        cache_hits = 0

        semaphore = asyncio.Semaphore(max_concurrent)
        total_papers = len(papers)

        async def process_with_limit(paper: PaperMetadata, index: int) -> dict:
            async with semaphore:
                if index > 0:
                    await asyncio.sleep(ACQUISITION_DELAY)
                return await acquire_and_process_single_paper(
                    paper, client, output_dir, index + 1, total_papers
                )

        tasks = [process_with_limit(paper, i) for i, paper in enumerate(papers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Paper pipeline task failed: {result}")
                continue

            doi = result.get("doi")
            if result.get("from_cache"):
                cache_hits += 1

            if result.get("acquired"):
                acquired[doi] = result.get("local_path")

                if result.get("processing_success"):
                    processing_results[doi] = result.get("processing_result")
                else:
                    processing_failed.append(doi)
            else:
                acquisition_failed.append(doi)

        logger.info(
            f"Paper pipeline complete: {len(acquired)} acquired "
            f"({cache_hits} from cache), "
            f"{len(processing_results)} processed, "
            f"{len(acquisition_failed)} acquisition failed, "
            f"{len(processing_failed)} processing failed"
        )
        return acquired, processing_results, acquisition_failed, processing_failed
