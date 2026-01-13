"""Paper acquisition module."""

from .core import (
    check_cache_for_paper,
    acquire_full_text,
    run_paper_pipeline,
)
from .sources import try_oa_download
from .types import (
    MAX_PROCESSING_CONCURRENT,
    PROCESSING_QUEUE_SIZE,
    OA_DOWNLOAD_TIMEOUT,
)

__all__ = [
    "check_cache_for_paper",
    "acquire_full_text",
    "run_paper_pipeline",
    "try_oa_download",
    "MAX_PROCESSING_CONCURRENT",
    "PROCESSING_QUEUE_SIZE",
    "OA_DOWNLOAD_TIMEOUT",
]
