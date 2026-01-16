"""Type definitions for the enhancement supervision workflow."""

from typing import Any, Literal, Optional

from typing_extensions import NotRequired, TypedDict


class EnhanceInput(TypedDict):
    """Input for report enhancement workflow."""

    report: str  # Markdown report to enhance
    topic: str
    research_questions: list[str]
    quality: Literal["quick", "standard", "comprehensive", "high_quality"]
    paper_corpus: NotRequired[dict[str, Any]]
    paper_summaries: NotRequired[dict[str, Any]]
    zotero_keys: NotRequired[dict[str, str]]


class EnhanceState(TypedDict, total=False):
    """State for enhancement orchestration."""

    # Core review state
    current_review: str
    final_review: Optional[str]
    review_loop1: Optional[str]
    review_loop2: Optional[str]

    # Paper corpus (accumulated)
    paper_corpus: dict[str, Any]
    paper_summaries: dict[str, Any]
    zotero_keys: dict[str, str]

    # Configuration
    input: EnhanceInput
    quality_settings: dict[str, Any]
    max_iterations_per_loop: int

    # Progress tracking
    loop_progress: list[dict]
    loop1_result: Optional[dict]
    loop2_result: Optional[dict]

    # Completion state
    completion_reason: str
    is_complete: bool
    errors: list[dict]


class EnhanceResult(TypedDict):
    """Result from enhancement workflow."""

    final_report: str
    review_loop1: Optional[str]
    review_loop2: Optional[str]
    loops_run: list[str]
    paper_corpus: dict[str, Any]
    paper_summaries: dict[str, Any]
    zotero_keys: dict[str, str]
    completion_reason: str
    errors: list[dict]
