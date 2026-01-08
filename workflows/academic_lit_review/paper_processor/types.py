"""Type definitions and constants for paper processing."""

from typing_extensions import TypedDict

from workflows.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    PaperSummary,
    QualitySettings,
)

MAX_PAPER_PIPELINE_CONCURRENT = 2
ACQUISITION_TIMEOUT = 300.0
RETRY_DELAY = 5.0
ACQUISITION_DELAY = 2.0


class PaperProcessingState(TypedDict):
    """State for paper processing subgraph."""

    input: LitReviewInput
    quality_settings: QualitySettings
    papers_to_process: list[PaperMetadata]
    acquired_papers: dict[str, str]
    acquisition_failed: list[str]
    processing_results: dict[str, dict]
    processing_failed: list[str]
    paper_summaries: dict[str, PaperSummary]
    elasticsearch_ids: dict[str, str]
    zotero_keys: dict[str, str]
