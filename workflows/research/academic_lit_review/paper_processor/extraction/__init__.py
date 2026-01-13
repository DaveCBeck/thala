"""Paper summary extraction from full text and metadata.

Uses unified structured output interface that auto-selects batch API for 5+ papers.
"""

from .core import (
    extract_paper_summary,
    extract_summary_from_metadata,
    extract_all_summaries,
)
from .parsers import (
    _generate_l2_from_l0,
    _fetch_content_for_extraction,
)
from .prompts import (
    L0_SIZE_THRESHOLD_FOR_L2,
    PAPER_SUMMARY_EXTRACTION_SYSTEM,
    METADATA_SUMMARY_EXTRACTION_SYSTEM,
)
from .types import PaperSummarySchema

__all__ = [
    "extract_paper_summary",
    "extract_summary_from_metadata",
    "extract_all_summaries",
    "_generate_l2_from_l0",
    "_fetch_content_for_extraction",
    "L0_SIZE_THRESHOLD_FOR_L2",
    "PAPER_SUMMARY_EXTRACTION_SYSTEM",
    "METADATA_SUMMARY_EXTRACTION_SYSTEM",
    "PaperSummarySchema",
]
