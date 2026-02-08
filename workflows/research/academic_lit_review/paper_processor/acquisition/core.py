"""Core paper acquisition and processing pipeline.

Architecture: Two-Stage Streaming Pipeline
==========================================
Phase 1: Check cache for all papers (fast, parallel)
Phase 2+3+4: Streaming acquisition → marker → LLM

  acquisition_producer
          ↓
  marker_queue (unbounded - holds file paths or markdown text, not PDF bytes)
          ↓
  marker_consumer
      - PDF → process_document_smart() → markdown (CPU or GPU path)
      - Already markdown → passthrough
          ↓
  llm_queue (unbounded)
          ↓
  llm_consumer
      - process_single_document(markdown_text)
          ↓
  results

Key insight: Both queues can be unbounded because they hold file path strings
or markdown text (~100KB-1MB), not PDF bytes in memory. The Marker service
has its own Redis/Celery queue that handles job queuing and backpressure.

This architecture ensures:
- Smart routing sends simple PDFs to fast CPU path, complex to GPU
- Marker GPU stays busy during batch API polling delays
- Processing starts as soon as first paper downloads

Fallback Mechanism:
- When papers fail (PDF validation, metadata mismatch, acquisition failure),
  a FallbackManager can provide alternative papers to process
- Fallback papers are injected into a retry queue for the acquisition producer
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from core.stores.retrieve_academic import RetrieveAcademicClient
from workflows.research.academic_lit_review.state import (
    FallbackSubstitution,
    PaperMetadata,
)
from workflows.research.academic_lit_review.paper_processor.cache import (
    check_document_exists_by_doi,
)
from workflows.research.academic_lit_review.paper_processor.document_processing import (
    process_single_document,
)
from workflows.research.academic_lit_review.paper_processor.types import (
    ACQUISITION_DELAY,
    ACQUISITION_TIMEOUT,
    MAX_PAPER_PIPELINE_CONCURRENT,
)

from core.scraping.pdf import process_document_smart, MarkerProcessingError
from core.scraping.types import ContentSource

from .sources import try_oa_download

if TYPE_CHECKING:
    from workflows.research.academic_lit_review.paper_processor.fallback import (
        FallbackManager,
    )

logger = logging.getLogger(__name__)


def _acq_log(doi: str, status: str, detail: str = "") -> None:
    """Emit the single structured ==ACQUISITION== log line for a paper."""
    parts = ["==ACQUISITION==", doi, status]
    if detail:
        parts.append(detail)
    logger.info("==".join(parts))


def _emit_oa_acq_log(
    doi: str,
    oa_url: str,
    content_source: Optional[ContentSource],
    provider: Optional[str],
) -> None:
    """Emit ==ACQUISITION== log for an OA download based on content source."""
    if content_source in (ContentSource.PDF_DIRECT, ContentSource.PDF_EXTRACTED):
        route = provider.removeprefix("smart:") if provider else "unknown"
        _acq_log(doi, "downloaded & processed open access pdf", route)
    else:
        _acq_log(doi, "downloaded open access html", oa_url)


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
        logger.debug(f"Acquired full text for {doi}: {saved_path}")
        return saved_path

    except Exception as e:
        logger.debug(f"Failed to acquire full text for {doi}: {e}")
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
            _acq_log(doi, "cached")
            logger.debug(f"Cache hit: {doi}")
        else:
            papers_to_acquire.append(papers_by_doi[doi])

    return cached_results, papers_to_acquire


async def run_paper_pipeline(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_PAPER_PIPELINE_CONCURRENT,
    fallback_manager: Optional["FallbackManager"] = None,
    checkpoint_callback: Optional[Callable[[int, dict], None]] = None,
    checkpoint_interval: int = 5,
) -> tuple[dict[str, str], dict[str, dict], list[str], list[str], list[FallbackSubstitution]]:
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

    Fallback mechanism: When papers fail (invalid PDF, acquisition failure),
    the fallback_manager provides alternative papers which are injected into
    a retry queue for processing.

    Checkpointing: When checkpoint_callback is provided, saves progress every
    checkpoint_interval papers to enable resumption after interruption.

    Args:
        papers: Papers to process
        max_concurrent: Maximum concurrent acquisitions (default: 2)
        fallback_manager: Optional manager for paper substitution on failure
        checkpoint_callback: Optional callback(iteration_count, partial_results) for mid-phase checkpointing
        checkpoint_interval: How often to checkpoint (default: every 5 papers)

    Returns:
        Tuple of (acquired, processing_results, acquisition_failed, processing_failed, fallback_substitutions)
    """
    async with RetrieveAcademicClient() as client:
        if not await client.health_check():
            logger.warning("Retrieve-academic service unavailable, skipping full-text acquisition")
            return {}, {}, [p.get("doi") for p in papers], [], []

        output_dir = Path("/tmp/thala_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        total_papers = len(papers)
        papers_by_doi = {p.get("doi"): p for p in papers}

        # Fallback retry queue - papers injected after failures
        retry_queue: asyncio.Queue = asyncio.Queue()

        # ========================================
        # Phase 1: Check cache for all papers
        # ========================================
        logger.debug(f"Phase 1: Checking cache for {total_papers} papers")
        cached_results, papers_to_acquire = await _check_cache_phase(papers)

        logger.info(f"Cache check complete: {len(cached_results)} cached, {len(papers_to_acquire)} need acquisition")

        if not papers_to_acquire:
            return {}, cached_results, [], [], []

        # ========================================
        # Phase 2+3: Streaming acquire → process
        # ========================================
        logger.debug(f"Phase 2+3: Starting streaming acquisition and processing for {len(papers_to_acquire)} papers")

        # Shared state (thread-safe via single event loop)
        acquired_paths: dict[str, str] = {}
        acquisition_failed: list[str] = []
        processing_results: dict[str, dict] = dict(cached_results)
        processing_failed: list[str] = []
        retrieve_academic_dois: set[str] = set()  # Track source for marker_consumer logging

        # Two-stage pipeline queues (both unbounded - hold file paths/markdown, not PDF bytes)
        marker_queue: asyncio.Queue = asyncio.Queue()
        llm_queue: asyncio.Queue = asyncio.Queue()

        # Coordination: acquisition_producer must wait for marker tasks to complete
        # before checking retry_queue, otherwise fallbacks injected by marker_consumer
        # can be missed (race condition).
        _marker_pending = 0
        _marker_idle = asyncio.Event()
        _marker_idle.set()  # Starts idle (no items yet)

        # Counters for progress logging
        acquired_count = 0
        processed_count = 0
        total_to_acquire = len(papers_to_acquire)
        last_checkpoint_count = 0  # Track last checkpoint for interval-based saving

        async def _send_to_marker(item: tuple) -> None:
            """Put item into marker_queue and track for idle coordination."""
            nonlocal _marker_pending
            _marker_pending += 1
            _marker_idle.clear()
            await marker_queue.put(item)

        async def acquisition_producer():
            """Submit jobs with rate limiting, poll completions, push to queue."""
            nonlocal acquired_count

            semaphore = asyncio.Semaphore(max_concurrent)

            # Track OA vs retrieve-academic acquisitions
            oa_acquired_count = 0

            async def try_acquire_single(paper: PaperMetadata, index: int) -> tuple[str, Optional[str], Optional[str]]:
                """Try OA first, then submit to retrieve-academic if needed.

                OA successes are pushed directly to the processing queue for
                immediate marker processing (streaming), rather than waiting
                for all OA attempts to complete.

                Returns:
                    (doi, job_id_or_none, local_path)
                    - If OA succeeded: (doi, None, None) - already pushed to queue
                    - If needs retrieve: (doi, job_id, local_path)
                    - If failed: raises exception
                """
                nonlocal oa_acquired_count, acquired_count

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
                        source, is_markdown, content_source, provider = await try_oa_download(oa_url, local_path, doi)
                        if source:
                            # Push directly to queue - stream to marker immediately
                            oa_acquired_count += 1
                            acquired_count += 1
                            acquired_paths[doi] = source
                            _emit_oa_acq_log(doi, oa_url, content_source, provider)
                            logger.debug(f"[{acquired_count}/{total_to_acquire}] Acquired via OA (streaming): {doi}")
                            await _send_to_marker((doi, source, papers_by_doi[doi], is_markdown))
                            return doi, None, None  # Signal OA handled

                    # Fall back to retrieve-academic
                    authors = [a.get("name") for a in paper.get("authors", [])[:5] if a.get("name")]
                    job = await client.retrieve(
                        doi=doi,
                        title=title,
                        authors=authors,
                        timeout_seconds=int(ACQUISITION_TIMEOUT),
                    )

                    return doi, job.job_id, str(local_path)

            try:
                # Try OA and submit retrieve jobs with rate limiting
                # OA successes are pushed to queue immediately (streaming)
                submit_tasks = [try_acquire_single(p, i) for i, p in enumerate(papers_to_acquire)]
                submit_results = await asyncio.gather(*submit_tasks, return_exceptions=True)

                # Collect retrieve-academic jobs for polling
                # (OA successes already pushed to queue in try_acquire_single)
                valid_jobs = []
                for i, result in enumerate(submit_results):
                    if isinstance(result, Exception):
                        doi = papers_to_acquire[i].get("doi")
                        logger.debug(f"Failed to acquire {doi}: {result}")

                        # Try to get a fallback paper
                        if fallback_manager:
                            fallback_paper = fallback_manager.get_fallback_for(doi, "acquisition_failed", "acquisition")
                            if fallback_paper:
                                # Add fallback to retry queue
                                fallback_doi = fallback_paper.get("doi")
                                papers_by_doi[fallback_doi] = fallback_paper
                                await retry_queue.put(fallback_paper)
                                _acq_log(doi, "could not download", f"fallback -> {fallback_doi}")
                                logger.debug(
                                    f"Fallback requested for {doi} -> {fallback_doi} (reason: acquisition_failed)"
                                )
                            else:
                                acquisition_failed.append(doi)
                                _acq_log(doi, "could not download", str(result))
                        else:
                            acquisition_failed.append(doi)
                            _acq_log(doi, "could not download", str(result))
                    else:
                        doi, job_id, local_path = result
                        if job_id is not None:
                            # Needs retrieve-academic polling
                            valid_jobs.append((doi, job_id, local_path))
                        # else: OA success, already pushed to queue

                logger.debug(
                    f"Acquired {oa_acquired_count} via OA (streamed to marker), "
                    f"submitted {len(valid_jobs)} to retrieve-academic"
                )

                # Poll retrieve-academic jobs for completions
                if valid_jobs:
                    async for (
                        doi,
                        local_path,
                        result,
                    ) in client.poll_jobs_until_complete(
                        valid_jobs,
                        poll_interval=2.0,
                        timeout=ACQUISITION_TIMEOUT,
                    ):
                        if isinstance(result, Exception):
                            logger.debug(f"Acquisition failed for {doi}: {result}")

                            # Try to get a fallback paper
                            if fallback_manager:
                                fallback_paper = fallback_manager.get_fallback_for(
                                    doi, "acquisition_failed", "acquisition"
                                )
                                if fallback_paper:
                                    fallback_doi = fallback_paper.get("doi")
                                    papers_by_doi[fallback_doi] = fallback_paper
                                    await retry_queue.put(fallback_paper)
                                    _acq_log(doi, "could not download", f"fallback -> {fallback_doi}")
                                    logger.debug(
                                        f"Fallback requested for {doi} -> {fallback_doi} (reason: acquisition_failed)"
                                    )
                                else:
                                    acquisition_failed.append(doi)
                                    _acq_log(doi, "could not download", str(result))
                            else:
                                acquisition_failed.append(doi)
                                _acq_log(doi, "could not download", str(result))
                        else:
                            acquired_paths[doi] = local_path
                            acquired_count += 1
                            retrieve_academic_dois.add(doi)
                            logger.debug(f"[{acquired_count}/{total_to_acquire}] Acquired: {doi}")
                            await _send_to_marker((doi, local_path, papers_by_doi[doi], False))

                # Process retry queue items (fallback papers from marker failures).
                # Wait for marker to finish processing before checking, to avoid
                # a race where marker_consumer injects fallbacks after we've
                # already checked the queue and sent the shutdown sentinel.
                while True:
                    await _marker_idle.wait()
                    if retry_queue.empty():
                        break

                    while not retry_queue.empty():
                        try:
                            fallback_paper = retry_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                        fallback_doi = fallback_paper.get("doi")
                        oa_url = fallback_paper.get("oa_url")
                        safe_doi = fallback_doi.replace("/", "_").replace(":", "_")
                        local_path = output_dir / f"{safe_doi}.pdf"

                        # Try OA download first
                        if oa_url:
                            source, is_markdown, content_source, provider = await try_oa_download(oa_url, local_path, fallback_doi)
                            if source:
                                acquired_paths[fallback_doi] = source
                                _emit_oa_acq_log(fallback_doi, oa_url, content_source, provider)
                                await _send_to_marker((fallback_doi, source, fallback_paper, is_markdown))
                                logger.debug(f"Fallback acquired via OA: {fallback_doi}")
                                continue

                        # Fall back to retrieve-academic
                        authors = [a.get("name") for a in fallback_paper.get("authors", [])[:5] if a.get("name")]
                        try:
                            path, _ = await client.retrieve_and_download(
                                doi=fallback_doi,
                                local_path=str(local_path),
                                title=fallback_paper.get("title", "Unknown"),
                                authors=authors,
                                timeout=ACQUISITION_TIMEOUT,
                            )
                            acquired_paths[fallback_doi] = path
                            retrieve_academic_dois.add(fallback_doi)
                            await _send_to_marker((fallback_doi, path, fallback_paper, False))
                            logger.debug(f"Fallback acquired via retrieve-academic: {fallback_doi}")
                        except Exception as e:
                            logger.debug(f"Fallback acquisition failed for {fallback_doi}: {e}")
                            acquisition_failed.append(fallback_doi)
                            _acq_log(fallback_doi, "could not download", str(e))

            finally:
                # Always signal end of acquisitions to marker stage
                await marker_queue.put(None)

        async def marker_consumer():
            """Stage 1: Convert PDFs to markdown, passthrough already-markdown items.

            Uses smart routing to send simple PDFs to fast CPU path (PyMuPDF)
            and complex/scanned PDFs to GPU path (Marker). The Marker service
            has its own Redis/Celery queue for GPU jobs.
            """
            active_tasks: set[asyncio.Task] = set()
            marker_completed = 0

            async def process_marker_item(doi: str, source: str, paper: PaperMetadata, is_markdown: bool):
                """Process a single item through marker stage."""
                nonlocal marker_completed
                try:
                    if is_markdown:
                        # Already markdown - passthrough to LLM stage
                        markdown_text = source
                        logger.debug(f"Marker passthrough (already markdown): {doi}")
                    else:
                        # PDF - convert via smart routing (CPU or GPU path)
                        logger.debug(f"Processing PDF via smart routing: {doi}")
                        pdf_bytes = Path(source).read_bytes()
                        result = await process_document_smart(pdf_bytes)
                        markdown_text = result.markdown
                        logger.debug(f"PDF processed via {result.processing_path}: {doi} ({len(markdown_text)} chars)")
                        if doi in retrieve_academic_dois:
                            _acq_log(doi, "downloaded & processed via retrieve-academic", result.processing_path)

                    marker_completed += 1
                    # Push markdown to LLM stage
                    await llm_queue.put((doi, markdown_text, paper))

                except MarkerProcessingError as e:
                    error_str = str(e)
                    # Determine failure reason for fallback
                    if "not a valid PDF" in error_str:
                        failure_reason = "pdf_invalid"
                    else:
                        failure_reason = "marker_error"

                    # Try to get a fallback paper
                    if fallback_manager:
                        fallback_paper = fallback_manager.get_fallback_for(doi, failure_reason, "marker")
                        if fallback_paper:
                            # Inject fallback into retry queue for acquisition
                            fallback_doi = fallback_paper.get("doi")
                            papers_by_doi[fallback_doi] = fallback_paper
                            await retry_queue.put(fallback_paper)
                            _acq_log(doi, "downloaded, not processed", f"fallback -> {fallback_doi}: {e}")
                            logger.debug(f"Fallback requested for {doi} -> {fallback_doi} (reason: {failure_reason})")
                        else:
                            # No fallback available - expected when queue exhausted
                            processing_failed.append(doi)
                            _acq_log(doi, "downloaded, not processed", str(e))
                            logger.debug(f"Marker error for {doi} (no fallback available): {e}")
                    else:
                        # No fallback manager
                        processing_failed.append(doi)
                        _acq_log(doi, "downloaded, not processed", str(e))
                        logger.debug(f"Marker error for {doi}: {e}")

                except Exception as e:
                    # Other marker failure - log error, add to failed
                    processing_failed.append(doi)
                    _acq_log(doi, "downloaded, not processed", f"{type(e).__name__}: {e}")
                    logger.error(
                        f"Marker error for {doi}: {type(e).__name__}: {e}",
                        exc_info=True,
                    )

            def _on_marker_task_done(task: asyncio.Task) -> None:
                """Track marker task completion for idle coordination."""
                nonlocal _marker_pending
                active_tasks.discard(task)
                _marker_pending -= 1
                if _marker_pending == 0:
                    _marker_idle.set()

            while True:
                item = await marker_queue.get()

                if item is None:
                    # End signal - wait for active tasks, then signal LLM stage
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                    await llm_queue.put(None)
                    logger.debug(f"Marker stage complete: {marker_completed} items processed")
                    break

                doi, source, paper, is_markdown = item

                # Create marker task
                task = asyncio.create_task(process_marker_item(doi, source, paper, is_markdown))
                active_tasks.add(task)
                task.add_done_callback(_on_marker_task_done)

        async def llm_consumer():
            """Stage 2: Process documents through document processing workflow.

            All papers run concurrently — batch API polling (30-60 min) is the
            bottleneck, and papers just await during that time with negligible
            resource cost. The marker queue provides natural backpressure (papers
            only arrive here after PDF→markdown conversion). Each paper gets its
            own broker batch group via contextvars, so concurrent papers are
            isolated. The broker handles rate limit backoff on direct API calls.

            Follows the same create_task + done_callback pattern as marker_consumer.
            """
            nonlocal processed_count, last_checkpoint_count
            active_tasks: set[asyncio.Task] = set()

            async def process_llm_item(doi: str, markdown_text: str, paper: PaperMetadata):
                """Process a single paper through the LLM pipeline."""
                nonlocal processed_count, last_checkpoint_count
                try:
                    result = await process_single_document(doi, markdown_text, paper, is_markdown=True)
                    processed_count += 1

                    if result.get("success"):
                        processing_results[doi] = result
                        logger.debug(f"[{processed_count}/{total_to_acquire}] LLM complete: {doi}")
                    elif result.get("validation_failed"):
                        processing_failed.append(doi)
                        logger.debug(
                            f"[{processed_count}/{total_to_acquire}] Validation failed for {doi}: "
                            f"{result.get('validation_reasoning', 'Unknown reason')}"
                        )
                        result["failure_reason"] = "metadata_mismatch"
                        result["failure_stage"] = "validation"
                    else:
                        processing_failed.append(doi)
                        logger.debug(f"[{processed_count}/{total_to_acquire}] LLM failed: {doi}")

                    # Checkpoint every N papers
                    if checkpoint_callback and processed_count - last_checkpoint_count >= checkpoint_interval:
                        last_checkpoint_count = processed_count
                        checkpoint_callback(processed_count, dict(processing_results))
                        logger.debug(f"Checkpoint: {processed_count}/{total_to_acquire} papers processed")

                except Exception as e:
                    logger.error(f"LLM consumer error for {doi}: {e}")
                    processed_count += 1
                    processing_failed.append(doi)

            while True:
                item = await llm_queue.get()
                if item is None:
                    # Wait for all in-flight tasks before shutting down
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                    logger.debug("LLM consumer: received shutdown signal")
                    break

                doi, markdown_text, paper = item
                task = asyncio.create_task(process_llm_item(doi, markdown_text, paper))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            logger.debug(f"LLM stage complete: {processed_count} items processed")

        # Run all three stages concurrently
        await asyncio.gather(
            acquisition_producer(),
            marker_consumer(),
            llm_consumer(),
        )

        # Collect fallback substitutions
        fallback_substitutions = fallback_manager.get_substitutions() if fallback_manager else []

        # Final summary
        logger.info(
            f"Paper pipeline complete: {len(acquired_paths)} acquired "
            f"({len(cached_results)} from cache), "
            f"{len(processing_results)} processed successfully, "
            f"{len(acquisition_failed)} acquisition failed, "
            f"{len(processing_failed)} processing failed, "
            f"{len(fallback_substitutions)} fallback substitutions"
        )

        return (
            acquired_paths,
            processing_results,
            acquisition_failed,
            processing_failed,
            fallback_substitutions,
        )
