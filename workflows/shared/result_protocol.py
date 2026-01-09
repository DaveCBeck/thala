"""
Standard result protocol for workflow APIs.

All workflows should include these common fields in their return values
to enable consistent handling by wrapper workflows.
"""

from typing import Literal, Optional
from typing_extensions import TypedDict


class WorkflowResultProtocol(TypedDict, total=False):
    """Common fields all workflow results must have.

    All workflow APIs should return a dict/TypedDict that includes these
    standard fields. Domain-specific fields can be added alongside these.

    Required fields (every workflow must set these):
        final_report: The main output text (report, review, markdown, etc.)
        status: Overall workflow status
        errors: List of errors encountered

    Optional common fields:
        source_count: Number of sources/papers/books used
        langsmith_run_id: LangSmith trace ID for debugging
        started_at: ISO timestamp when workflow started
        completed_at: ISO timestamp when workflow completed
    """

    # Required fields - every workflow must set these
    final_report: str  # Unified name for main output
    status: Literal["success", "partial", "failed"]
    errors: list[dict]  # [{phase: str, error: str, timestamp?: str}]

    # Optional common fields
    source_count: int  # Number of sources used
    langsmith_run_id: Optional[str]
    started_at: str  # ISO timestamp
    completed_at: str  # ISO timestamp


# Type alias for quality tier literals used across workflows
WebResearchQuality = Literal["quick", "standard", "comprehensive"]
AcademicQuality = Literal["test", "quick", "standard", "comprehensive", "high_quality"]
BookFindingQuality = Literal["quick", "standard", "comprehensive"]
