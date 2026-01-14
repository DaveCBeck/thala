"""State schema for substack_review workflow."""

from operator import add
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict


class EssayDraft(TypedDict):
    """A single essay draft from a writing agent."""

    angle: Literal["puzzle", "finding", "contrarian"]
    content: str
    word_count: int


class EssayInput(TypedDict):
    """Input for the workflow."""

    literature_review: str  # Raw markdown with [@KEY] citations


class FormattedReference(TypedDict):
    """A formatted reference from Zotero lookup."""

    key: str
    citation_text: str
    found_in_zotero: bool


class SubstackReviewState(TypedDict):
    """Main workflow state for lit review to substack transformation."""

    # Input
    input: EssayInput

    # Validation phase
    is_valid: bool
    validation_error: Optional[str]
    extracted_citation_keys: list[str]  # All [@KEY] citations found in input

    # Writing phase (parallel aggregation via add reducer)
    essay_drafts: Annotated[list[EssayDraft], add]

    # Selection phase
    selected_angle: Optional[Literal["puzzle", "finding", "contrarian"]]
    selection_reasoning: Optional[str]
    essay_evaluations: Optional[dict]  # Per-essay strength/weakness from chooser

    # Reference formatting phase
    formatted_references: list[FormattedReference]
    missing_references: list[str]  # Keys not found in Zotero

    # Output
    final_essay: Optional[str]  # Winning essay + formatted references

    # Workflow metadata
    status: Optional[Literal["success", "partial", "failed"]]
    errors: Annotated[list[dict], add]
