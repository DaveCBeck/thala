"""
Type definitions for supervision loop.

Pydantic schemas for structured LLM outputs (supervisor decisions).
TypedDict state types are imported from the main state module.
"""

import json
import re
from typing import Literal, Optional, Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Re-export TypedDict types from main state module
from workflows.academic_lit_review.state import (
    SupervisionState,
    SupervisionExpansion,
)

__all__ = [
    "IdentifiedIssue",
    "SupervisorDecision",
    "SupervisionState",
    "SupervisionExpansion",
    "MAX_SUPERVISION_DEPTH",
    "LiteratureBase",
    "LiteratureBaseDecision",
    "ArchitecturalAssessment",
    "StructuralEdit",
    "EditManifest",
    "ArchitectureVerificationResult",
    "SectionEditResult",
    "HolisticReviewResult",
    "CohesionCheckResult",
    "Edit",
    "DocumentEdits",
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
# Loop 2: Literature Base Expansion
# =============================================================================

class LiteratureBase(BaseModel):
    """A missing literature base identified by Opus."""
    name: str = Field(description="Name of the literature base (e.g., 'health economics')")
    rationale: str = Field(description="Why this literature is important for the review")
    perspective_type: Literal["supportive", "challenging", "analogous"] = Field(
        description="Whether this literature supports, challenges, or provides analogy to the argument"
    )
    search_queries: list[str] = Field(description="Queries to discover this literature")
    integration_strategy: str = Field(description="How to integrate findings into the review")

class LiteratureBaseDecision(BaseModel):
    """Decision from Loop 2 analyzer."""
    action: Literal["expand_base", "pass_through"] = Field(
        description="Whether to expand a new literature base or pass through"
    )
    literature_base: Optional[LiteratureBase] = Field(
        default=None, description="The literature base to expand (only if action is expand_base)"
    )
    reasoning: str = Field(description="Academic justification for the decision")

# =============================================================================
# Loop 3: Structure and Cohesion
# =============================================================================


class ArchitecturalAssessment(BaseModel):
    """Assessment of document information architecture."""
    section_organization_score: float = Field(
        ge=0.0, le=1.0,
        description="How well sections are organized (1.0 = excellent)"
    )
    content_placement_issues: list[str] = Field(
        default_factory=list,
        description="Content in wrong sections (e.g., 'methodology in introduction')"
    )
    logical_flow_issues: list[str] = Field(
        default_factory=list,
        description="Breaks in argument flow or logical jumps"
    )
    anti_patterns_detected: list[str] = Field(
        default_factory=list,
        description="Structural anti-patterns (sprawl, premature detail, orphaned content, redundant framing)"
    )


class StructuralEdit(BaseModel):
    """A single structural edit in the manifest."""
    edit_type: Literal[
        "reorder_sections",   # Move paragraph to new position
        "merge_sections",     # Combine two paragraphs
        "add_transition",     # Insert transitional text
        "move_content",       # Relocate content from source to target section
        "delete_paragraph",   # Remove truly redundant paragraph
        "trim_redundancy",    # Remove redundant portion while keeping essential content
        "split_section",      # Split one section into multiple
    ] = Field(description="Type of structural edit")
    source_paragraph: int = Field(ge=1, description="Paragraph number to act on (must be >= 1)")
    target_paragraph: Optional[int] = Field(default=None, ge=1, description="Target paragraph for reorder/merge/add_transition/move_content")
    content_to_preserve: Optional[str] = Field(
        default=None,
        description="For move_content: specific content to relocate (if not entire paragraph)"
    )
    replacement_text: Optional[str] = Field(
        default=None,
        description="For trim_redundancy: replacement text. For split_section: text with ---SPLIT--- delimiter"
    )
    notes: str = Field(min_length=1, description="Explanation of why this edit improves the document")

    @model_validator(mode='after')
    def validate_target_required(self) -> 'StructuralEdit':
        """Ensure target_paragraph is provided for edit types that require it."""
        if self.edit_type in ("reorder_sections", "merge_sections", "add_transition", "move_content"):
            if self.target_paragraph is None:
                raise ValueError(f"{self.edit_type} requires target_paragraph to be specified")
            if self.source_paragraph == self.target_paragraph:
                raise ValueError(f"source_paragraph and target_paragraph cannot be the same ({self.source_paragraph})")
        return self

class EditManifest(BaseModel):
    """Complete edit manifest from Loop 3 Analyst."""
    architecture_assessment: Optional[ArchitecturalAssessment] = Field(
        default=None,
        description="Document architecture analysis (section organization, content placement, flow)"
    )
    edits: list[StructuralEdit] = Field(default_factory=list, description="List of structural edits")
    todo_markers: list[str] = Field(default_factory=list, description="<!-- TODO: ... --> markers to insert")
    overall_assessment: str = Field(min_length=10, description="Summary of structural issues found")
    needs_restructuring: bool = Field(description="Whether document needs restructuring")

    @field_validator("edits", mode="before")
    @classmethod
    def validate_edits_list(cls, v: Any) -> list:
        """Ensure edits is a proper list, surface parsing failures."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            raise ValueError(f"edits must be a list, got string: {v[:100]}...")
        return v if v is not None else []

    @model_validator(mode='after')
    def validate_consistency(self) -> 'EditManifest':
        """Enforce that needs_restructuring=True has corresponding edits.

        Raises ValueError to trigger LLM retry if the constraint is violated.
        The prompt explicitly states this is INVALID and will be rejected.
        """
        if self.needs_restructuring and len(self.edits) == 0 and len(self.todo_markers) == 0:
            raise ValueError(
                "CONSTRAINT VIOLATION: needs_restructuring=True requires at least one edit or todo_marker. "
                "If you identified issues but cannot determine specific fixes, set needs_restructuring=False. "
                "Do not flag issues you cannot concretely fix."
            )
        return self


class ArchitectureVerificationResult(BaseModel):
    """Result from post-edit architecture verification."""
    issues_resolved: list[str] = Field(
        default_factory=list,
        description="List of original issues that are now resolved"
    )
    issues_remaining: list[str] = Field(
        default_factory=list,
        description="List of issues still present after edits"
    )
    regressions_introduced: list[str] = Field(
        default_factory=list,
        description="New issues introduced by the edits"
    )
    coherence_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall coherence of document after edits (1.0 = fully coherent)"
    )
    needs_another_iteration: bool = Field(
        description="Whether another editing iteration is needed"
    )
    reasoning: str = Field(description="Explanation of verification findings")

# =============================================================================
# Loop 4: Section-Level Deep Editing
# =============================================================================

class SectionEditResult(BaseModel):
    """Output from a parallel section editor."""
    section_id: str = Field(description="Identifier of the edited section")
    edited_content: str = Field(description="The edited section content")
    notes: str = Field(description="Cross-references and suggested edits for other sections")
    new_paper_todos: list[str] = Field(default_factory=list, description="TODOs for potential new papers (as a JSON array of strings)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the edits")

    @field_validator("new_paper_todos", mode="before")
    @classmethod
    def parse_string_todos(cls, v: Any) -> list[str]:
        """Handle LLM returning a numbered list or prose string instead of JSON array."""
        if isinstance(v, str):
            if not v.strip():
                return []
            # Try parsing as JSON first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
            # Fall back to splitting by newlines and cleaning up numbered list format
            lines = v.strip().split("\n")
            items = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Remove common numbered list prefixes: "1.", "1)", "- ", "* "
                cleaned = re.sub(r'^(\d+[\.\)]\s*|[-*]\s*)', '', line).strip()
                if cleaned:
                    items.append(cleaned)
            return items if items else [v]  # Return original as single item if no structure found
        return v if v is not None else []

class HolisticReviewResult(BaseModel):
    """Output from Phase B holistic reviewer."""
    sections_approved: list[str] = Field(default_factory=list, description="Section IDs that are approved")
    sections_flagged: list[str] = Field(default_factory=list, description="Section IDs that need re-editing")
    flagged_reasons: dict[str, str] = Field(default_factory=dict, description="Reasons for flagging each section")
    overall_coherence_score: float = Field(ge=0.0, le=1.0, description="Overall document coherence score")

# =============================================================================
# Loop 4.5: Cohesion Check
# =============================================================================

class CohesionCheckResult(BaseModel):
    """Structured output from Loop 4.5 cohesion check."""
    needs_restructuring: bool = Field(description="Whether document needs to return to Loop 3")
    reasoning: str = Field(description="Explanation of the assessment")

# =============================================================================
# Loop 5: Fact and Reference Checking
# =============================================================================

class Edit(BaseModel):
    """A single fact/reference edit."""
    find: str = Field(description="Exact text to find (must be unique in document)")
    replace: str = Field(description="Replacement text")
    edit_type: Literal["fact_correction", "citation_fix", "clarity"] = Field(
        description="Type of edit being made"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this edit")
    source_doi: Optional[str] = Field(default=None, description="DOI of paper supporting this edit")

class DocumentEdits(BaseModel):
    """Collection of edits from fact/reference checker."""
    edits: list[Edit] = Field(default_factory=list, description="List of edits to apply")
    reasoning: str = Field(default="", description="Overall reasoning for the edits")
    ambiguous_claims: list[str] = Field(default_factory=list, description="Claims needing human review")
    unaddressed_todos: list[str] = Field(default_factory=list, description="TODOs that couldn't be resolved")

    @field_validator("ambiguous_claims", "unaddressed_todos", mode="before")
    @classmethod
    def parse_json_strings(cls, v: Any) -> list[str]:
        """Handle LLM returning JSON string instead of list."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            # If not valid JSON list, return as single-item list
            return [v] if v.strip() else []
        return v if v is not None else []


# =============================================================================
# Constants
# =============================================================================

MAX_SUPERVISION_DEPTH = 2  # Hard limit on recursive supervision
