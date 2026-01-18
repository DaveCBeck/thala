"""Pydantic schemas for structured LLM outputs in editing workflow."""

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


# =============================================================================
# Structural Analysis Schemas (Phase 2)
# =============================================================================


class StructuralIssue(BaseModel):
    """A structural problem identified in the document."""

    issue_id: str = Field(description="Unique issue identifier (e.g., 'issue_1')")

    issue_type: Literal[
        # Content organization issues
        "content_sprawl",  # Topic scattered across 3+ non-adjacent sections
        "misplaced_content",  # Content in wrong section for its topic
        "orphaned_content",  # Content disconnected from surroundings
        # Structural completeness issues
        "missing_introduction",  # Document or major section lacks intro
        "missing_conclusion",  # Document or major section lacks conclusion
        "missing_synthesis",  # Thematic section lacks synthesizing discussion
        "abrupt_ending",  # Document ends mid-thought or incompletely
        # Redundancy issues
        "duplicate_framing",  # Multiple intros or conclusions
        "redundant_content",  # Same points made in multiple places
        "overlapping_sections",  # Two sections cover same ground
        # Flow issues
        "logical_gap",  # Argument jumps without connection
        "missing_transition",  # Abrupt shift between sections
        "broken_narrative",  # Story/argument thread lost
    ] = Field(description="Category of structural issue")

    severity: Literal["minor", "moderate", "major", "critical"] = Field(
        description="Impact on document coherence"
    )

    # References use stable IDs from document model
    affected_section_ids: list[str] = Field(
        description="Section IDs involved in this issue"
    )
    affected_block_ids: list[str] = Field(
        default_factory=list,
        description="Specific block IDs if issue is localized",
    )

    description: str = Field(
        min_length=20, description="Detailed description of the problem"
    )

    recommended_action: Literal[
        "move_section",  # Relocate entire section
        "consolidate",  # Merge scattered content
        "generate_intro",  # Add introduction
        "generate_conclusion",  # Add conclusion
        "generate_synthesis",  # Add synthesizing discussion
        "generate_transition",  # Add transitional content
        "delete_redundant",  # Remove duplicate content
        "merge_sections",  # Combine overlapping sections
        "split_section",  # Break apart overloaded section
        "reorder_within",  # Reorder content within section
    ] = Field(description="Recommended fix approach")

    action_details: str = Field(
        description="Specific guidance for implementing the fix"
    )


class StructuralAnalysis(BaseModel):
    """Complete structural analysis of document."""

    # Document-level assessment
    has_clear_introduction: bool = Field(
        description="Whether document has a clear introduction"
    )
    has_clear_conclusion: bool = Field(
        description="Whether document has a clear conclusion"
    )
    narrative_coherence_score: float = Field(
        ge=0.0, le=1.0, description="Overall narrative flow quality"
    )
    section_organization_score: float = Field(
        ge=0.0, le=1.0, description="How well sections are organized"
    )

    # Identified issues (ordered by severity)
    issues: list[StructuralIssue] = Field(default_factory=list)

    # Summary
    overall_assessment: str = Field(
        min_length=50, description="Summary of document structure quality"
    )

    @property
    def critical_issues_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def major_issues_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "major")

    @property
    def needs_structural_work(self) -> bool:
        return self.critical_issues_count > 0 or self.major_issues_count > 0


# =============================================================================
# Edit Operation Schemas (Phase 3-4)
# =============================================================================


class SectionMoveEdit(BaseModel):
    """Move a section to a new location."""

    edit_type: Literal["section_move"] = "section_move"
    source_section_id: str = Field(description="Section to move")
    target_position: Literal["before", "after"] = Field(
        description="Position relative to target"
    )
    target_section_id: str = Field(description="Reference section for placement")
    justification: str = Field(description="Why this move improves the document")


class SectionMergeEdit(BaseModel):
    """Merge two sections into one."""

    edit_type: Literal["section_merge"] = "section_merge"
    primary_section_id: str = Field(description="Section to keep/expand")
    secondary_section_id: str = Field(description="Section to merge in")
    merge_strategy: Literal["append", "interleave", "synthesize"] = Field(
        description="How to combine the sections"
    )
    new_heading: Optional[str] = Field(
        default=None, description="Optional new heading for merged section"
    )
    justification: str


class ContentConsolidationEdit(BaseModel):
    """Consolidate scattered content about a topic into one location."""

    edit_type: Literal["consolidate"] = "consolidate"
    topic: str = Field(description="What the scattered content is about")
    source_block_ids: list[str] = Field(
        description="Blocks to consolidate (from different sections)"
    )
    target_section_id: str = Field(
        description="Where to put consolidated content"
    )
    consolidation_approach: Literal["merge_and_dedupe", "synthesize", "reorganize"] = (
        Field(description="How to consolidate the content")
    )
    justification: str


class GenerateIntroductionEdit(BaseModel):
    """Generate missing introduction content."""

    edit_type: Literal["generate_introduction"] = "generate_introduction"
    scope: Literal["document", "section"] = Field(
        description="Whether for whole document or a section"
    )
    target_section_id: Optional[str] = Field(
        default=None, description="Section ID (None for document-level)"
    )
    context_section_ids: list[str] = Field(
        description="Sections to reference for context"
    )
    introduction_requirements: str = Field(
        description="What the intro should accomplish"
    )
    target_word_count: int = Field(default=200, ge=50, le=500)


class GenerateConclusionEdit(BaseModel):
    """Generate missing conclusion content."""

    edit_type: Literal["generate_conclusion"] = "generate_conclusion"
    scope: Literal["document", "section"]
    target_section_id: Optional[str] = Field(
        default=None, description="Section ID to add conclusion to (for section scope)"
    )
    insert_after_section_id: Optional[str] = Field(
        default=None,
        description="Section ID after which to insert new conclusion section (for document scope)",
    )
    context_section_ids: list[str]
    conclusion_requirements: str
    target_word_count: int = Field(default=300, ge=100, le=800)
    new_section_heading: str = Field(
        default="Conclusion",
        description="Heading for new conclusion section (document scope)",
    )


class GenerateSynthesisEdit(BaseModel):
    """Generate synthesizing discussion for a thematic section."""

    edit_type: Literal["generate_synthesis"] = "generate_synthesis"
    target_section_id: str
    synthesis_requirements: str = Field(description="What should be synthesized")
    position: Literal["start", "end"] = Field(default="end")
    target_word_count: int = Field(default=400, ge=100, le=1000)


class GenerateTransitionEdit(BaseModel):
    """Generate transition between sections."""

    edit_type: Literal["generate_transition"] = "generate_transition"
    from_section_id: str
    to_section_id: str
    transition_type: Literal["bridging", "contrast", "progression", "pivot"] = Field(
        description="Type of transition needed"
    )
    target_word_count: int = Field(default=100, ge=30, le=300)


class DeleteRedundantEdit(BaseModel):
    """Delete redundant content blocks."""

    edit_type: Literal["delete_redundant"] = "delete_redundant"
    block_ids_to_delete: list[str]
    primary_block_id: str = Field(
        description="Block that adequately covers the content"
    )
    justification: str


class TrimRedundancyEdit(BaseModel):
    """Trim redundant portions from a block while keeping essential content."""

    edit_type: Literal["trim_redundancy"] = "trim_redundancy"
    block_id: str
    content_to_remove: str = Field(description="Specific redundant content")
    justification: str


# Union type for all edits
StructuralEdit = Union[
    SectionMoveEdit,
    SectionMergeEdit,
    ContentConsolidationEdit,
    GenerateIntroductionEdit,
    GenerateConclusionEdit,
    GenerateSynthesisEdit,
    GenerateTransitionEdit,
    DeleteRedundantEdit,
    TrimRedundancyEdit,
]


class EditPlan(BaseModel):
    """Ordered plan of edits to apply."""

    edits: list[
        Union[
            SectionMoveEdit,
            SectionMergeEdit,
            ContentConsolidationEdit,
            GenerateIntroductionEdit,
            GenerateConclusionEdit,
            GenerateSynthesisEdit,
            GenerateTransitionEdit,
            DeleteRedundantEdit,
            TrimRedundancyEdit,
        ]
    ] = Field(default_factory=list)

    execution_order_rationale: str = Field(
        description="Why edits are ordered this way"
    )
    estimated_word_count_change: int = Field(
        default=0, description="Estimated net change in word count"
    )

    @property
    def structure_edits(self) -> list:
        """Edits that change document structure."""
        structure_types = {"section_move", "section_merge", "consolidate"}
        return [e for e in self.edits if e.edit_type in structure_types]

    @property
    def generation_edits(self) -> list:
        """Edits that generate new content."""
        gen_types = {
            "generate_introduction",
            "generate_conclusion",
            "generate_synthesis",
            "generate_transition",
        }
        return [e for e in self.edits if e.edit_type in gen_types]

    @property
    def removal_edits(self) -> list:
        """Edits that remove content."""
        removal_types = {"delete_redundant", "trim_redundancy"}
        return [e for e in self.edits if e.edit_type in removal_types]


# =============================================================================
# Verification Schemas (Phase 5)
# =============================================================================


class StructureVerification(BaseModel):
    """Result of structure verification after edits."""

    coherence_score: float = Field(
        ge=0.0, le=1.0, description="Overall coherence after edits"
    )

    issues_resolved: list[str] = Field(
        default_factory=list, description="Issue IDs that are now fixed"
    )
    issues_remaining: list[str] = Field(
        default_factory=list, description="Issues that still need attention"
    )
    regressions: list[str] = Field(
        default_factory=list, description="New problems introduced by edits"
    )

    structure_improved: bool
    flow_improved: bool
    completeness_improved: bool

    reasoning: str = Field(description="Explanation of verification findings")

    @property
    def needs_another_iteration(self) -> bool:
        return self.coherence_score < 0.8 and (
            len(self.issues_remaining) > 0 or len(self.regressions) > 0
        )


# =============================================================================
# Enhancement Schemas (Phase 6 - when document has citations)
# =============================================================================


class SectionEnhancement(BaseModel):
    """Result of enhancing a single section."""

    section_id: str = Field(description="ID of the enhanced section")
    original_word_count: int = Field(description="Word count before enhancement")
    enhanced_word_count: int = Field(description="Word count after enhancement")
    enhanced_content: str = Field(description="The enhanced section content")

    citations_added: list[str] = Field(
        default_factory=list,
        description="New citation keys added ([@KEY] format)",
    )
    citations_removed: list[str] = Field(
        default_factory=list,
        description="Citation keys removed",
    )

    enhancement_notes: str = Field(
        description="Notes about what was enhanced and why"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the enhancement"
    )

    @property
    def word_count_change(self) -> int:
        return self.enhanced_word_count - self.original_word_count

    @property
    def word_count_change_percent(self) -> float:
        if self.original_word_count == 0:
            return 0.0
        return self.word_count_change / self.original_word_count


class EnhanceCoherenceReview(BaseModel):
    """Coherence review after an enhancement pass."""

    coherence_score: float = Field(
        ge=0.0, le=1.0, description="Overall coherence after enhancements"
    )
    sections_enhanced: list[str] = Field(
        description="Section IDs that were enhanced in this pass"
    )
    sections_needing_work: list[str] = Field(
        default_factory=list,
        description="Section IDs that need more enhancement",
    )
    issues_found: list[str] = Field(
        default_factory=list,
        description="Issues discovered during coherence review",
    )

    overall_assessment: str = Field(
        description="Summary of enhancement pass results"
    )

    @property
    def needs_another_iteration(self) -> bool:
        return self.coherence_score < 0.75 and len(self.sections_needing_work) > 0


# Note: Fact-check and reference-check schemas are now in
# workflows.enhance.fact_check.schemas


# =============================================================================
# Polish Schemas (Phase 7)
# =============================================================================


class PolishScreeningResult(BaseModel):
    """Result of screening sections for polish work."""

    sections_to_polish: list[str] = Field(
        default_factory=list,
        description="Section IDs that need flow/clarity improvements",
    )
    sections_ok: list[str] = Field(
        default_factory=list,
        description="Section IDs that are already well-polished",
    )
    screening_summary: str = Field(
        default="Screening incomplete",
        description="Brief summary of polish needs",
    )


class SectionPolish(BaseModel):
    """Result of polishing a single section."""

    polished_content: str = Field(
        description="The polished section content with improved flow and clarity"
    )
    changes_made: list[str] = Field(
        default_factory=list,
        description="List of specific improvements made",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        default=0.8,
        description="Confidence in the polish quality",
    )


# =============================================================================
# Final Verification (Phase 9)
# =============================================================================


class FinalVerification(BaseModel):
    """Final quality check after all edits."""

    coherence_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    flow_score: float = Field(ge=0.0, le=1.0)

    has_introduction: bool
    has_conclusion: bool
    sections_well_organized: bool

    remaining_issues: list[str] = Field(
        default_factory=list, description="Any issues that couldn't be resolved"
    )

    overall_assessment: str

    @property
    def overall_score(self) -> float:
        return (
            self.coherence_score + self.completeness_score + self.flow_score
        ) / 3.0
