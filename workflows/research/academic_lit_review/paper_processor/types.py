"""Type definitions and constants for paper processing."""

from typing import Optional
from typing_extensions import TypedDict

from workflows.research.academic_lit_review.state import (
    FallbackCandidate,
    FallbackSubstitution,
    LitReviewInput,
    PaperMetadata,
    PaperSummary,
    QualitySettings,
)
from workflows.shared.language import LanguageConfig

MAX_PAPER_PIPELINE_CONCURRENT = 2
ACQUISITION_TIMEOUT = 300.0
RETRY_DELAY = 5.0
ACQUISITION_DELAY = 2.0


class LanguageVerificationStats(TypedDict, total=False):
    """Statistics from language verification."""

    verified_count: int  # Papers that passed verification
    rejected_count: int  # Papers rejected for wrong language
    skipped_count: int  # Papers skipped (no content for detection)
    by_detected_language: dict[str, int]  # Count by detected language


class PaperProcessingState(TypedDict, total=False):
    """State for paper processing subgraph."""

    # Required
    input: LitReviewInput
    quality_settings: QualitySettings
    papers_to_process: list[PaperMetadata]

    # Acquisition results
    acquired_papers: dict[str, str]
    acquisition_failed: list[str]
    processing_results: dict[str, dict]
    processing_failed: list[str]

    # Language verification (optional - only used for non-English)
    language_config: Optional[LanguageConfig]
    language_rejected_dois: list[str]  # DOIs rejected for wrong language
    language_verification_stats: LanguageVerificationStats

    # Fallback mechanism
    fallback_queue: list[FallbackCandidate]  # Ordered by relevance (overflow first, then near-threshold)
    fallback_substitutions: list[FallbackSubstitution]  # Track all substitutions made
    fallback_exhausted: list[str]  # DOIs that failed with no fallback available

    # Final outputs
    paper_summaries: dict[str, PaperSummary]
    elasticsearch_ids: dict[str, str]
    zotero_keys: dict[str, str]
