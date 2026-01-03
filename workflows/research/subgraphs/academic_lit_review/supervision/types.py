"""
Type definitions for supervision loop.

Pydantic schemas for structured LLM outputs (supervisor decisions).
TypedDict state types are imported from the main state module.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Re-export TypedDict types from main state module
from workflows.research.subgraphs.academic_lit_review.state import (
    SupervisionState,
    SupervisionExpansion,
)

__all__ = [
    "IdentifiedIssue",
    "SupervisorDecision",
    "SupervisionState",
    "SupervisionExpansion",
    "MAX_SUPERVISION_DEPTH",
]


# =============================================================================
# Pydantic Schemas for Structured LLM Outputs
# =============================================================================


class IdentifiedIssue(BaseModel):
    """Issue identified by supervisor - focused on theoretical depth."""

    topic: str = Field(
        description="Specific theory, concept, or foundational element that needs exploration"
    )
    issue_type: Literal[
        "underlying_theory",        # Core theoretical framework missing
        "methodological_foundation", # Research methodology not grounded
        "unifying_thread",          # Key argument connecting themes
        "foundational_concept",     # Background concept assumed but unexplained
    ] = Field(
        description="Category of the identified gap"
    )
    rationale: str = Field(
        description="Why addressing this strengthens the paper academically"
    )
    research_query: str = Field(
        description="Search query to use for discovering relevant literature"
    )
    related_section: str = Field(
        description="Which section of the review this issue relates to"
    )
    integration_guidance: str = Field(
        description="How the findings should be integrated into the review"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this is a genuine theoretical gap worth exploring"
    )


class SupervisorDecision(BaseModel):
    """Decision output from the supervisor analysis."""

    action: Literal["research_needed", "pass_through"] = Field(
        description="Whether additional research is needed or the review is theoretically sound"
    )
    issue: Optional[IdentifiedIssue] = Field(
        default=None,
        description="The identified issue to explore (only if action is research_needed)"
    )
    reasoning: str = Field(
        description="Academic justification for the decision"
    )


# =============================================================================
# Constants
# =============================================================================

MAX_SUPERVISION_DEPTH = 2  # Hard limit on recursive supervision
