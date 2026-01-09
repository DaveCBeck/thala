"""
Shared result types for workflow wrappers.

These types define the standard interface for workflow results
used by both multi_lang and wrapped orchestration workflows.
"""

from datetime import datetime
from typing import Literal, Optional
from typing_extensions import TypedDict


class WorkflowResult(TypedDict):
    """Standard result format for workflow adapters in registry-based invocation.

    This is the minimal result type returned by workflow adapters.
    Used by multi_lang workflow registry.
    """

    final_report: str | None  # The main text/markdown output
    source_count: int  # Number of sources found/processed
    status: str  # "success" | "partial" | "failed"
    errors: list[dict]  # List of {phase, error} dicts


class WrapperResult(TypedDict):
    """Extended result type for wrapped workflow orchestration.

    Includes timing and workflow identification information.
    Used by wrapped workflow nodes.
    """

    workflow_type: str  # "web_research" | "academic_lit_review" | "book_finding" | etc.
    final_output: Optional[str]  # The markdown/report content
    started_at: datetime
    completed_at: Optional[datetime]
    status: Literal["pending", "running", "completed", "failed"]
    error: Optional[str]
    top_of_mind_id: Optional[str]  # UUID after saving to store
