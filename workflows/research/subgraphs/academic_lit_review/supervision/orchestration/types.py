"""Types for orchestration state."""
from typing import Optional, Any
from typing_extensions import TypedDict

from ..types import (
    LiteratureBaseDecision,
    EditManifest,
    HolisticReviewResult,
    CohesionCheckResult,
    DocumentEdits,
)
from workflows.research.subgraphs.academic_lit_review.state import (
    LitReviewInput,
    QualitySettings,
    PaperSummary,
    ThematicCluster,
    MultiLoopProgress,
)


class OrchestrationState(TypedDict, total=False):
    """State for multi-loop supervision orchestration."""

    # Review content
    current_review: str
    final_review: Optional[str]

    # Context from main workflow
    input: LitReviewInput
    paper_corpus: dict[str, Any]
    paper_summaries: dict[str, PaperSummary]
    clusters: list[ThematicCluster]
    quality_settings: QualitySettings
    zotero_keys: dict[str, str]

    # Multi-loop tracking
    loop_progress: MultiLoopProgress
    loop3_repeat_count: int  # For Loop 4.5 -> Loop 3 return (max = max_iterations_per_loop)

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
