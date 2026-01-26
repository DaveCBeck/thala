"""State schema for evening_reads workflow.

Transforms academic literature reviews into a 4-part series:
1 overview + 3 deep-dives.
"""

from operator import add
from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict


class EveningReadsInput(TypedDict):
    """Input for the workflow."""

    literature_review: str  # Raw markdown with [@KEY] citations


class CitationKeyMapping(TypedDict):
    """Mapping from Zotero key to ES record information."""

    zotero_key: str
    es_record_id: Optional[str]  # UUID string, None if not found in store
    title: Optional[str]


class DeepDiveAssignment(TypedDict):
    """Assignment for a single deep-dive article."""

    id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"]
    title: str  # Evocative title for the article
    theme: str  # Brief description of the theme
    structural_approach: Literal["puzzle", "finding", "contrarian"]  # Narrative approach
    anchor_keys: list[str]  # 2-3 Zotero keys that anchor this deep-dive
    relevant_sections: list[str]  # Section headers from lit review to focus on


class EnrichedContent(TypedDict):
    """Content fetched from store for a deep-dive."""

    deep_dive_id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"]
    zotero_key: str
    content: str  # L2 or L0 content
    content_level: Literal["L0", "L2"]  # Which compression level was used


class DeepDiveDraft(TypedDict):
    """A single deep-dive draft."""

    id: Literal["deep_dive_1", "deep_dive_2", "deep_dive_3"]
    title: str
    content: str
    word_count: int
    citation_keys: list[str]  # Citations used in this draft


class OverviewDraft(TypedDict):
    """The overview article draft."""

    title: str
    content: str
    word_count: int
    citation_keys: list[str]


class FormattedReference(TypedDict):
    """A formatted reference from Zotero lookup."""

    key: str
    citation_text: str
    found_in_zotero: bool


class FinalOutput(TypedDict):
    """Final formatted output for a single article."""

    id: Literal["overview", "deep_dive_1", "deep_dive_2", "deep_dive_3"]
    title: str
    content: str  # Article with references appended
    word_count: int


class ImageOutput(TypedDict):
    """Generated header image for an article."""

    article_id: Literal["overview", "deep_dive_1", "deep_dive_2", "deep_dive_3"]
    image_bytes: bytes
    prompt_used: str


class EveningReadsState(TypedDict):
    """Main workflow state for lit review to evening reads series."""

    # Input
    input: EveningReadsInput

    # Validation phase
    is_valid: bool
    validation_error: Optional[str]
    extracted_citation_keys: list[str]  # All [@KEY] citations found in input
    citation_mappings: dict[str, CitationKeyMapping]  # zotero_key -> mapping

    # Planning phase
    deep_dive_assignments: list[DeepDiveAssignment]  # 3 planned deep-dives
    overview_scope: str  # Description of what the overview should cover

    # Fetching phase (parallel aggregation via add reducer)
    enriched_content: Annotated[list[EnrichedContent], add]

    # Writing phase (parallel aggregation via add reducer)
    deep_dive_drafts: Annotated[list[DeepDiveDraft], add]
    overview_draft: Optional[OverviewDraft]

    # Reference formatting phase
    formatted_references: list[FormattedReference]
    missing_references: list[str]

    # Final output
    final_outputs: list[FinalOutput]  # All 4 articles with references

    # Image generation
    image_outputs: Annotated[list[ImageOutput], add]

    # Workflow metadata
    status: Optional[Literal["success", "partial", "failed"]]
    errors: Annotated[list[dict], add]
