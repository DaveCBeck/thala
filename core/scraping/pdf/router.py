"""Document routing - coordinates CPU and GPU processing paths.

This module provides smart routing between CPU (PyMuPDF) and GPU (Marker) paths
based on document complexity analysis.
"""

import asyncio
import atexit
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from .analysis import DocumentAnalysis, analyze_document
from .cpu_extractor import extract_text_cpu
from .metrics import (
    record_analysis_time,
    record_cpu_fallback,
    record_processing_time,
    record_route_decision,
)
from .processor import check_marker_available, process_pdf_bytes
from .routing import RouteDecision, determine_route

logger = logging.getLogger(__name__)

# Thread pool for parallel document analysis (CPU-bound PyMuPDF operations)
_analysis_executor: ThreadPoolExecutor | None = None

# Cached Marker availability status (None = not checked, True/False = checked)
_marker_available: bool | None = None
_marker_check_lock: asyncio.Lock | None = None


def _get_marker_check_lock() -> asyncio.Lock:
    """Get or create the Marker check lock."""
    global _marker_check_lock
    if _marker_check_lock is None:
        _marker_check_lock = asyncio.Lock()
    return _marker_check_lock


async def check_marker_for_session(force_recheck: bool = False) -> bool:
    """Check Marker availability once per session, caching the result.

    This avoids per-document network calls when Marker is down. The check
    is done once and cached for the session. Use force_recheck=True to
    re-check (e.g., after Marker is restarted).

    Args:
        force_recheck: If True, re-check even if already cached

    Returns:
        True if Marker is available, False otherwise
    """
    global _marker_available

    # Fast path: already checked
    if _marker_available is not None and not force_recheck:
        return _marker_available

    # Serialize checks to avoid multiple concurrent health checks
    async with _get_marker_check_lock():
        # Double-check after acquiring lock
        if _marker_available is not None and not force_recheck:
            return _marker_available

        _marker_available = await check_marker_available()

        if not _marker_available:
            logger.warning(
                "Marker service unavailable - using CPU-only mode. "
                "Complex/scanned PDFs will have degraded quality. "
                "Restart Marker and call reset_marker_session_cache() to re-enable GPU processing."
            )
        else:
            logger.info("Marker service available - GPU processing enabled")

        return _marker_available


def reset_marker_session_cache() -> None:
    """Reset the cached Marker availability status.

    Call this after Marker is restarted to re-enable GPU processing.
    """
    global _marker_available
    _marker_available = None
    logger.info("Marker session cache reset - will re-check on next document")


def _get_analysis_executor() -> ThreadPoolExecutor:
    """Get or create the analysis thread pool."""
    global _analysis_executor
    if _analysis_executor is None:
        _analysis_executor = ThreadPoolExecutor(
            max_workers=16,
            thread_name_prefix="pdf_analysis_",
        )
    return _analysis_executor


def _cleanup_analysis_executor():
    """Shutdown executor on process exit."""
    if _analysis_executor is not None:
        _analysis_executor.shutdown(wait=False)


atexit.register(_cleanup_analysis_executor)


class ProcessingResult(NamedTuple):
    """Unified result from either CPU or GPU processing."""

    markdown: str
    page_count: int
    processing_path: str  # "cpu", "gpu", or "gpu_fallback"
    analysis: DocumentAnalysis | None
    route_decision: RouteDecision | None


async def process_document_smart(
    pdf_content: bytes,
    *,
    force_gpu: bool = False,
    quality: str | None = None,
) -> ProcessingResult:
    """Route document to optimal processing path.

    Uses cached Marker availability check to avoid per-document network calls.
    When Marker is unavailable, forces CPU-only mode for graceful degradation.

    Args:
        pdf_content: Raw PDF bytes
        force_gpu: Always use Marker (for quality-critical documents)
        quality: Override quality preset (if None, auto-selects based on complexity)

    Returns:
        ProcessingResult with markdown and metadata
    """
    # Step 1: Check Marker availability (cached, no network call after first check)
    marker_ok = await check_marker_for_session()

    # Step 2: Analyze document in thread pool (CPU-bound PyMuPDF operation)
    loop = asyncio.get_running_loop()
    analysis_start = time.perf_counter()
    analysis = await loop.run_in_executor(
        _get_analysis_executor(),
        analyze_document,
        pdf_content,
    )
    analysis_duration = time.perf_counter() - analysis_start
    record_analysis_time(analysis_duration)

    # Step 3: Make routing decision (separate from analysis)
    route = determine_route(analysis)

    # Auto-select quality preset based on complexity if not specified
    effective_quality = quality or _quality_for_complexity(analysis)

    # Step 4: Handle Marker unavailable - force CPU-only mode
    if not marker_ok:
        logger.info(
            f"Document analysis: {analysis.complexity.value}, "
            f"pages={analysis.page_count}, scanned={analysis.is_scanned}, "
            f"route=cpu_degraded (Marker unavailable)"
        )

        processing_start = time.perf_counter()
        result = await extract_text_cpu(pdf_content)
        processing_duration = time.perf_counter() - processing_start

        record_route_decision("cpu_degraded", analysis.complexity.value, analysis.page_count)
        record_processing_time("cpu_degraded", processing_duration)

        # Log warning for documents that really needed GPU
        if analysis.is_scanned:
            logger.warning(
                f"Scanned PDF processed via CPU (Marker unavailable) - "
                f"quality may be poor. {analysis.page_count} pages."
            )
        elif route.queue == "gpu":
            logger.info(
                f"Complex PDF processed via CPU (Marker unavailable) - "
                f"quality may be degraded. Complexity: {analysis.complexity.value}"
            )

        return ProcessingResult(
            markdown=result.markdown,
            page_count=result.page_count,
            processing_path="cpu_degraded",
            analysis=analysis,
            route_decision=route,
        )

    # Normal routing with Marker available
    logger.info(
        f"Document analysis: {analysis.complexity.value}, "
        f"pages={analysis.page_count}, images={analysis.has_images}, "
        f"tables={analysis.has_tables}, scanned={analysis.is_scanned}, "
        f"route={route.queue}, confidence={route.confidence:.2f}, "
        f"quality={effective_quality}"
    )

    processing_start = time.perf_counter()

    # Route based on decision
    if force_gpu or route.queue == "gpu":
        # Full Marker pipeline
        markdown = await process_pdf_bytes(
            pdf_content,
            quality=effective_quality,
        )

        processing_duration = time.perf_counter() - processing_start
        record_route_decision("gpu", analysis.complexity.value, analysis.page_count)
        record_processing_time("gpu", processing_duration)

        return ProcessingResult(
            markdown=markdown,
            page_count=analysis.page_count,
            processing_path="gpu",
            analysis=analysis,
            route_decision=route,
        )

    # CPU fast-path
    result = await extract_text_cpu(pdf_content)

    # Check if CPU extraction flagged issues (double-check during extraction)
    if result.fallback_recommended:
        logger.info(f"CPU extraction confidence {result.confidence:.2f} below threshold, falling back to GPU")
        record_cpu_fallback()

        # Fall back to GPU
        markdown = await process_pdf_bytes(
            pdf_content,
            quality=effective_quality,
        )

        processing_duration = time.perf_counter() - processing_start
        record_route_decision("gpu_fallback", analysis.complexity.value, analysis.page_count)
        record_processing_time("gpu_fallback", processing_duration)

        return ProcessingResult(
            markdown=markdown,
            page_count=analysis.page_count,
            processing_path="gpu_fallback",
            analysis=analysis,
            route_decision=route,
        )

    processing_duration = time.perf_counter() - processing_start
    record_route_decision("cpu", analysis.complexity.value, analysis.page_count)
    record_processing_time("cpu", processing_duration)

    return ProcessingResult(
        markdown=result.markdown,
        page_count=result.page_count,
        processing_path="cpu",
        analysis=analysis,
        route_decision=route,
    )


async def process_document_smart_url(
    url: str,
    *,
    force_gpu: bool = False,
    quality: str | None = None,
    timeout: float = 60.0,
) -> ProcessingResult:
    """Download PDF from URL and route to optimal processing path.

    Uses httpx with Playwright fallback for sites that block direct downloads.

    Args:
        url: URL to PDF file
        force_gpu: Always use Marker (for quality-critical documents)
        quality: Override quality preset (if None, auto-selects based on complexity)
        timeout: Download timeout in seconds

    Returns:
        ProcessingResult with markdown and metadata
    """
    from .processor import _download_pdf

    logger.debug(f"Downloading PDF from URL for smart routing: {url}")
    pdf_content = await _download_pdf(url, timeout=timeout)
    logger.debug(f"Downloaded PDF: {len(pdf_content) / 1024:.1f} KB")

    return await process_document_smart(
        pdf_content,
        force_gpu=force_gpu,
        quality=quality,
    )


def _quality_for_complexity(analysis: DocumentAnalysis) -> str:
    """Map document complexity to Marker quality preset.

    Quality presets already incorporate batch multipliers:
    - fast: batch_multiplier=4 (for simple, digital docs)
    - balanced: batch_multiplier=2 (general purpose)
    - quality: batch_multiplier=1 (for complex/scanned docs with tables)
    """
    from .analysis import DocumentComplexity

    return {
        DocumentComplexity.LIGHT: "fast",
        DocumentComplexity.MIXED: "balanced",
        DocumentComplexity.HEAVY: "quality",
    }.get(analysis.complexity, "balanced")
