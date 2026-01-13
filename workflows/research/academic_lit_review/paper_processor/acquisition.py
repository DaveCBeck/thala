"""Paper acquisition and processing pipeline.

DEPRECATED: This module has been refactored into acquisition/ subfolder.
Kept for backward compatibility - all imports are re-exported from the new structure.

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

from .acquisition import (
    check_cache_for_paper,
    acquire_full_text,
    run_paper_pipeline,
    try_oa_download,
    _is_pdf_url,
    _download_pdf_from_url,
    MAX_PROCESSING_CONCURRENT,
    PROCESSING_QUEUE_SIZE,
    OA_DOWNLOAD_TIMEOUT,
)

__all__ = [
    "check_cache_for_paper",
    "acquire_full_text",
    "run_paper_pipeline",
    "try_oa_download",
    "_is_pdf_url",
    "_download_pdf_from_url",
    "MAX_PROCESSING_CONCURRENT",
    "PROCESSING_QUEUE_SIZE",
    "OA_DOWNLOAD_TIMEOUT",
]
