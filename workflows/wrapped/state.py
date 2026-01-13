"""
State schemas for wrapped research workflow.

Defines TypedDict states for orchestrating web research, academic lit review,
and book finding with unified quality settings.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from workflows.shared.wrappers.quality import QualityTier


# =============================================================================
# Quality Configuration
# =============================================================================


# Import shared quality mapping but provide backwards-compatible names
# The shared mapping uses workflow keys, we alias to legacy names for this workflow
def _build_legacy_quality_mapping() -> dict[str, dict[str, str]]:
    """Build legacy quality mapping from shared infrastructure."""
    from workflows.shared.wrappers.quality import QUALITY_MAPPING as SHARED_MAPPING

    legacy_mapping = {}
    for tier, workflow_qualities in SHARED_MAPPING.items():
        legacy_mapping[tier] = {
            "web_depth": workflow_qualities.get("web_research", tier),
            "academic_quality": workflow_qualities.get("academic_lit_review", tier),
            "book_quality": workflow_qualities.get("book_finding", tier),
        }
    return legacy_mapping


QUALITY_MAPPING: dict[str, dict[str, str]] = _build_legacy_quality_mapping()


# =============================================================================
# Input Types
# =============================================================================


class WrappedResearchInput(TypedDict):
    """Input parameters for wrapped research workflow."""

    query: str
    quality: QualityTier
    research_questions: Optional[list[str]]  # For academic lit review
    date_range: Optional[tuple[int, int]]  # For academic lit review


# =============================================================================
# Workflow Result Types
# =============================================================================


class WorkflowResult(TypedDict):
    """Result from a single sub-workflow."""

    workflow_type: str  # "web" | "academic" | "books"
    final_output: Optional[str]  # The markdown/report content
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # "pending" | "running" | "completed" | "failed"
    error: Optional[str]
    top_of_mind_id: Optional[str]  # UUID after saving to top_of_mind


# =============================================================================
# Main State
# =============================================================================


class WrappedResearchState(TypedDict):
    """Complete state for wrapped research orchestration."""

    # Input
    input: WrappedResearchInput

    # Workflow results
    web_result: Optional[WorkflowResult]
    academic_result: Optional[WorkflowResult]
    book_result: Optional[WorkflowResult]

    # Intermediate: book finding query generated from research
    book_theme: Optional[str]  # Generated theme for book finding
    book_brief: Optional[str]  # Context for book search

    # Final outputs
    combined_summary: Optional[str]  # LLM-generated synthesis of all three
    top_of_mind_ids: dict[str, str]  # {workflow_type: uuid}

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str
    langsmith_run_id: Optional[str]  # For tracing in LangSmith
    errors: Annotated[list[dict], add]
