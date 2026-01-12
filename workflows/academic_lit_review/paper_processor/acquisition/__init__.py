"""Paper acquisition module.

Re-exports for backward compatibility with existing imports.
"""

from .core import (
    check_cache_for_paper,
    acquire_full_text,
    run_paper_pipeline,
)
from .sources import try_oa_download
from .http_client import _is_pdf_url, _download_pdf_from_url
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
    "_is_pdf_url",
    "_download_pdf_from_url",
    "MAX_PROCESSING_CONCURRENT",
    "PROCESSING_QUEUE_SIZE",
    "OA_DOWNLOAD_TIMEOUT",
]
