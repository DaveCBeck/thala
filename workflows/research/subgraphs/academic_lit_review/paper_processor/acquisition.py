"""Paper acquisition and processing pipeline.

Architecture: Phased Pipeline
=============================
Phase 1: Check cache for all papers (fast, parallel)
Phase 2: Acquire all non-cached PDFs (rate-limited for external APIs)
Phase 3: Process all acquired PDFs (parallel, fills marker queue)

This separation ensures:
- Marker's Celery queue stays full (workers never idle)
- External API rate limiting is independent of processing speed
- Post-processing starts as soon as any document completes
"""

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

# Processing concurrency can be higher than pipeline concurrency
# since marker has its own queue and multiple workers
MAX_PROCESSING_CONCURRENT = 4


async def check_cache_for_paper(paper: PaperMetadata) -> tuple[str, Optional[dict]]:
    """Check if paper exists in cache.

    Returns:
        Tuple of (doi, cached_result or None)
    """
    doi = paper.get("doi")
    existing = await check_document_exists_by_doi(doi)
    if existing:
        return doi, {
            "doi": doi,
            "success": True,
            "es_record_id": existing["es_record_id"],
            "zotero_key": existing["zotero_key"],
            "short_summary": existing["short_summary"],
            "errors": [],
        }
    return doi, None


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


async def run_paper_pipeline(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_PAPER_PIPELINE_CONCURRENT,
) -> tuple[dict[str, str], dict[str, dict], list[str], list[str]]:
    """Run phased acquire→process pipeline for all papers.

    Phase 1: Check cache for all papers (parallel, fast)
    Phase 2: Acquire all non-cached PDFs (rate-limited)
    Phase 3: Process all acquired PDFs (parallel, fills marker queue)

    Args:
        papers: Papers to process
        max_concurrent: Maximum concurrent acquisitions (default: 2)

    Returns:
        Tuple of (acquired, processing_results, acquisition_failed, processing_failed)
    """
    async with RetrieveAcademicClient() as client:
        if not await client.health_check():
            logger.warning("Retrieve-academic service unavailable, skipping full-text acquisition")
            return {}, {}, [p.get("doi") for p in papers], []

        output_dir = Path("/tmp/thala_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        total_papers = len(papers)
        papers_by_doi = {p.get("doi"): p for p in papers}

        # ========================================
        # Phase 1: Check cache for all papers
        # ========================================
        logger.info(f"Phase 1: Checking cache for {total_papers} papers...")
        cache_tasks = [check_cache_for_paper(p) for p in papers]
        cache_results = await asyncio.gather(*cache_tasks)

        cached_results = {}
        papers_to_acquire = []
        for doi, cached in cache_results:
            if cached:
                cached_results[doi] = cached
                logger.info(f"Cache hit: {doi}")
            else:
                papers_to_acquire.append(papers_by_doi[doi])

        logger.info(
            f"Phase 1 complete: {len(cached_results)} cached, "
            f"{len(papers_to_acquire)} need acquisition"
        )

        # ========================================
        # Phase 2: Acquire all non-cached PDFs
        # ========================================
        acquired_paths = {}  # doi -> local_path
        acquisition_failed = []

        if papers_to_acquire:
            logger.info(f"Phase 2: Acquiring {len(papers_to_acquire)} PDFs...")
            semaphore = asyncio.Semaphore(max_concurrent)

            async def acquire_with_limit(paper: PaperMetadata, index: int) -> tuple[str, Optional[str]]:
                async with semaphore:
                    if index > 0:
                        await asyncio.sleep(ACQUISITION_DELAY)
                    doi = paper.get("doi")
                    path = await acquire_full_text(paper, client, output_dir)
                    return doi, path

            acquire_tasks = [
                acquire_with_limit(p, i) for i, p in enumerate(papers_to_acquire)
            ]
            acquire_results = await asyncio.gather(*acquire_tasks, return_exceptions=True)

            for result in acquire_results:
                if isinstance(result, Exception):
                    logger.error(f"Acquisition task failed: {result}")
                    continue
                doi, path = result
                if path:
                    acquired_paths[doi] = path
                else:
                    acquisition_failed.append(doi)

            logger.info(
                f"Phase 2 complete: {len(acquired_paths)} acquired, "
                f"{len(acquisition_failed)} failed"
            )

        # ========================================
        # Phase 3: Process all acquired PDFs
        # ========================================
        processing_results = dict(cached_results)  # Start with cached
        processing_failed = []

        if acquired_paths:
            logger.info(f"Phase 3: Processing {len(acquired_paths)} documents...")
            process_semaphore = asyncio.Semaphore(MAX_PROCESSING_CONCURRENT)
            completed_count = 0
            total_to_process = len(acquired_paths)

            async def process_with_limit(doi: str, path: str) -> tuple[str, dict]:
                nonlocal completed_count
                async with process_semaphore:
                    paper = papers_by_doi[doi]
                    result = await process_single_document(doi, path, paper)
                    completed_count += 1
                    title = paper.get("title", "Unknown")[:50]
                    if result.get("success"):
                        logger.info(
                            f"[{completed_count}/{total_to_process}] Completed: {title}"
                        )
                    else:
                        logger.warning(
                            f"[{completed_count}/{total_to_process}] Failed: {title}"
                        )
                    return doi, result

            process_tasks = [
                process_with_limit(doi, path) for doi, path in acquired_paths.items()
            ]
            process_results = await asyncio.gather(*process_tasks, return_exceptions=True)

            for result in process_results:
                if isinstance(result, Exception):
                    logger.error(f"Processing task failed: {result}")
                    continue
                doi, proc_result = result
                if proc_result.get("success"):
                    processing_results[doi] = proc_result
                else:
                    processing_failed.append(doi)

            logger.info(
                f"Phase 3 complete: {len(processing_results) - len(cached_results)} processed, "
                f"{len(processing_failed)} failed"
            )

        # Final summary
        logger.info(
            f"Paper pipeline complete: {len(acquired_paths)} acquired "
            f"({len(cached_results)} from cache), "
            f"{len(processing_results)} processed, "
            f"{len(acquisition_failed)} acquisition failed, "
            f"{len(processing_failed)} processing failed"
        )

        return (
            acquired_paths,
            processing_results,
            acquisition_failed,
            processing_failed,
        )
