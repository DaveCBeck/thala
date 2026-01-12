"""Core paper acquisition and processing pipeline.

Architecture: Streaming Producer-Consumer Pipeline
==================================================
Phase 1: Check cache for all papers (fast, parallel)
Phase 2+3: Streaming acquisition → processing
  - Producer: Submit jobs with rate limiting, poll completions
  - Queue: Buffer acquired papers for processing (bounded for backpressure)
  - Consumer: Process papers as they arrive from queue

This architecture ensures:
- Processing starts as soon as first paper downloads (not after all)
- Marker GPU stays busy throughout the pipeline
- Memory bounded by queue size (~8 papers × 50MB = ~400MB)
- External API rate limiting is independent of processing speed
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from core.stores.retrieve_academic import RetrieveAcademicClient
from workflows.academic_lit_review.state import PaperMetadata
from workflows.academic_lit_review.paper_processor.cache import check_document_exists_by_doi
from workflows.academic_lit_review.paper_processor.document_processing import process_single_document
from workflows.academic_lit_review.paper_processor.types import (
    ACQUISITION_DELAY,
    ACQUISITION_TIMEOUT,
    MAX_PAPER_PIPELINE_CONCURRENT,
)

from .sources import try_oa_download
from .types import MAX_PROCESSING_CONCURRENT, PROCESSING_QUEUE_SIZE

logger = logging.getLogger(__name__)


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


async def _check_cache_phase(
    papers: list[PaperMetadata],
) -> tuple[dict[str, dict], list[PaperMetadata]]:
    """Phase 1: Check cache for all papers.

    Args:
        papers: Papers to check

    Returns:
        Tuple of (cached_results dict, papers_to_acquire list)
    """
    papers_by_doi = {p.get("doi"): p for p in papers}

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

    return cached_results, papers_to_acquire


async def run_paper_pipeline(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_PAPER_PIPELINE_CONCURRENT,
    use_batch_api: bool = True,
) -> tuple[dict[str, str], dict[str, dict], list[str], list[str]]:
    """Run streaming acquire→process pipeline for all papers.

    Architecture: Streaming Producer-Consumer Pipeline
    ==================================================
    Phase 1: Check cache for all papers (parallel, fast)
    Phase 2+3: Streaming acquisition → processing
      - Producer: Submit jobs with rate limiting, poll completions
      - Queue: Buffer acquired papers for processing
      - Consumer: Process papers as they arrive from queue

    This ensures:
    - Processing starts as soon as first paper downloads (not after all)
    - Marker GPU stays busy throughout the pipeline
    - Memory bounded by queue size (~8 papers)

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
        cached_results, papers_to_acquire = await _check_cache_phase(papers)

        logger.info(
            f"Phase 1 complete: {len(cached_results)} cached, "
            f"{len(papers_to_acquire)} need acquisition"
        )

        if not papers_to_acquire:
            return {}, cached_results, [], []

        # ========================================
        # Phase 2+3: Streaming acquire → process
        # ========================================
        logger.info(
            f"Phase 2+3: Streaming acquisition and processing for "
            f"{len(papers_to_acquire)} papers..."
        )

        # Shared state (thread-safe via single event loop)
        acquired_paths: dict[str, str] = {}
        acquisition_failed: list[str] = []
        processing_results: dict[str, dict] = dict(cached_results)
        processing_failed: list[str] = []

        # Queue bridges acquisition → processing with backpressure
        processing_queue: asyncio.Queue = asyncio.Queue(maxsize=PROCESSING_QUEUE_SIZE)

        # Counters for progress logging
        acquired_count = 0
        processed_count = 0
        total_to_acquire = len(papers_to_acquire)

        async def acquisition_producer():
            """Submit jobs with rate limiting, poll completions, push to queue."""
            nonlocal acquired_count

            semaphore = asyncio.Semaphore(max_concurrent)

            # Track OA vs retrieve-academic acquisitions
            oa_acquired_count = 0

            async def try_acquire_single(
                paper: PaperMetadata, index: int
            ) -> tuple[str, Optional[str], Optional[str], bool]:
                """Try OA first, then submit to retrieve-academic if needed.

                Returns:
                    (doi, job_id_or_none, local_path, is_markdown)
                    - If OA succeeded: (doi, None, source, is_markdown) - source is path or markdown
                    - If needs retrieve: (doi, job_id, local_path, False)
                    - If failed: raises exception
                """
                nonlocal oa_acquired_count

                async with semaphore:
                    if index > 0:
                        await asyncio.sleep(ACQUISITION_DELAY)

                    doi = paper.get("doi")
                    title = paper.get("title", "Unknown")
                    oa_url = paper.get("oa_url")

                    safe_doi = doi.replace("/", "_").replace(":", "_")
                    local_path = output_dir / f"{safe_doi}.pdf"

                    # Try OA download first if URL available
                    if oa_url:
                        source, is_markdown = await try_oa_download(oa_url, local_path, doi)
                        if source:
                            oa_acquired_count += 1
                            return doi, None, source, is_markdown

                    # Fall back to retrieve-academic
                    authors = [a.get("name") for a in paper.get("authors", [])[:5] if a.get("name")]
                    job = await client.retrieve(
                        doi=doi,
                        title=title,
                        authors=authors,
                        timeout_seconds=int(ACQUISITION_TIMEOUT),
                    )

                    return doi, job.job_id, str(local_path), False

            try:
                # Try OA and submit retrieve jobs with rate limiting
                submit_tasks = [
                    try_acquire_single(p, i) for i, p in enumerate(papers_to_acquire)
                ]
                submit_results = await asyncio.gather(*submit_tasks, return_exceptions=True)

                # Process results: OA successes go directly to queue, others need polling
                valid_jobs = []
                for i, result in enumerate(submit_results):
                    if isinstance(result, Exception):
                        doi = papers_to_acquire[i].get("doi")
                        logger.error(f"Failed to acquire {doi}: {result}")
                        acquisition_failed.append(doi)
                    else:
                        doi, job_id, source, is_markdown = result
                        if job_id is None:
                            # OA success - push directly to processing queue
                            acquired_paths[doi] = source
                            acquired_count += 1
                            logger.info(
                                f"[{acquired_count}/{total_to_acquire}] Acquired via OA: {doi}"
                            )
                            await processing_queue.put((doi, source, papers_by_doi[doi], is_markdown))
                        else:
                            # Needs retrieve-academic polling
                            valid_jobs.append((doi, job_id, source))

                logger.info(
                    f"Acquired {oa_acquired_count} via OA, "
                    f"submitted {len(valid_jobs)} to retrieve-academic"
                )

                # Poll retrieve-academic jobs for completions
                if valid_jobs:
                    async for doi, local_path, result in client.poll_jobs_until_complete(
                        valid_jobs,
                        poll_interval=2.0,
                        timeout=ACQUISITION_TIMEOUT,
                    ):
                        if isinstance(result, Exception):
                            acquisition_failed.append(doi)
                            logger.warning(f"Acquisition failed for {doi}: {result}")
                        else:
                            acquired_paths[doi] = local_path
                            acquired_count += 1
                            logger.info(
                                f"[{acquired_count}/{total_to_acquire}] Acquired: {doi}, "
                                f"queuing for processing"
                            )
                            await processing_queue.put((doi, local_path, papers_by_doi[doi], False))

            finally:
                # Always signal end of acquisitions
                await processing_queue.put(None)

        async def processing_consumer():
            """Process papers from queue as they arrive."""
            nonlocal processed_count

            process_semaphore = asyncio.Semaphore(MAX_PROCESSING_CONCURRENT)
            active_tasks: set[asyncio.Task] = set()

            async def process_item(
                doi: str, source: str, paper: PaperMetadata, is_markdown: bool
            ):
                """Process a single document with concurrency limiting."""
                nonlocal processed_count
                async with process_semaphore:
                    try:
                        result = await process_single_document(doi, source, paper, is_markdown, use_batch_api)
                        processed_count += 1
                        title = paper.get("title", "Unknown")[:50]

                        if result.get("success"):
                            processing_results[doi] = result
                            logger.info(
                                f"[{processed_count}/{total_to_acquire}] Processed: {title}"
                            )
                        else:
                            processing_failed.append(doi)
                            logger.warning(
                                f"[{processed_count}/{total_to_acquire}] Failed: {title}"
                            )
                    except Exception as e:
                        processing_failed.append(doi)
                        logger.error(f"Processing error for {doi}: {e}")

            while True:
                item = await processing_queue.get()

                if item is None:
                    # End signal - wait for active tasks to complete
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                    break

                doi, source, paper, is_markdown = item

                # Create processing task
                task = asyncio.create_task(process_item(doi, source, paper, is_markdown))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

        # Run producer and consumer concurrently
        await asyncio.gather(
            acquisition_producer(),
            processing_consumer(),
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
