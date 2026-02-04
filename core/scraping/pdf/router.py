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
from .processor import process_pdf_bytes
from .routing import RouteDecision, determine_route

logger = logging.getLogger(__name__)

# Thread pool for parallel document analysis (CPU-bound PyMuPDF operations)
_analysis_executor: ThreadPoolExecutor | None = None


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

    Args:
        pdf_content: Raw PDF bytes
        force_gpu: Always use Marker (for quality-critical documents)
        quality: Override quality preset (if None, auto-selects based on complexity)

    Returns:
        ProcessingResult with markdown and metadata
    """
    # Step 1: Analyze document in thread pool (CPU-bound PyMuPDF operation)
    loop = asyncio.get_running_loop()
    analysis_start = time.perf_counter()
    analysis = await loop.run_in_executor(
        _get_analysis_executor(),
        analyze_document,
        pdf_content,
    )
    analysis_duration = time.perf_counter() - analysis_start
    record_analysis_time(analysis_duration)

    # Step 2: Make routing decision (separate from analysis)
    route = determine_route(analysis)

    # Auto-select quality preset based on complexity if not specified
    effective_quality = quality or _quality_for_complexity(analysis)

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
