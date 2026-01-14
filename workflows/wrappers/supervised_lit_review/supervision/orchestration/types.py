"""Types for orchestration state."""

from typing import Optional, Any
from typing_extensions import TypedDict

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    QualitySettings,
    PaperSummary,
    ThematicCluster,
    MultiLoopProgress,
)


class ZoteroKeyMetadata(TypedDict, total=False):
    """Metadata about a Zotero citation key's origin and verification status."""

    key: str
    source: str  # CITATION_SOURCE_* constant
    verified: bool  # True if programmatically verified against Zotero
    doi: Optional[str]


class OrchestrationState(TypedDict, total=False):
    """State for multi-loop supervision orchestration."""

    # Review content
    current_review: str
    final_review: Optional[str]

    # Intermediate review snapshots (after each loop)
    review_loop1: Optional[str]
    review_loop2: Optional[str]
    review_loop3: Optional[str]
    review_loop4: Optional[str]

    # Context from main workflow
    input: LitReviewInput
    paper_corpus: dict[str, Any]
    paper_summaries: dict[str, PaperSummary]
    clusters: list[ThematicCluster]
    quality_settings: QualitySettings
    zotero_keys: dict[str, str]

    # Citation key source tracking (key -> ZoteroKeyMetadata)
    zotero_key_sources: dict[str, ZoteroKeyMetadata]

    # Multi-loop tracking
    loop_progress: MultiLoopProgress
    loop3_repeat_count: int

    # Outputs from each loop
    loop1_result: Optional[dict]
    loop2_result: Optional[dict]
    loop3_result: Optional[dict]
    loop4_result: Optional[dict]
    loop4_5_result: Optional[dict]
    loop5_result: Optional[dict]

    # Final outputs
    human_review_items: list[str]
    completion_reason: str
    is_complete: bool

    # Error tracking across loops
    loop_errors: list[dict]

    # Zotero verification settings
    verify_zotero: bool
