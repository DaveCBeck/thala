"""State schemas for synthesis workflow.

Defines TypedDict states for the multi-phase synthesis workflow that
orchestrates academic lit review, supervision, web research, book finding,
and editing into a comprehensive synthesis.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal, Optional

from typing_extensions import TypedDict

from workflows.shared.quality_config import QualityTier


# =============================================================================
# Core Types
# =============================================================================


class MultiLangConfig(TypedDict, total=False):
    """Configuration for multi-language research in synthesis."""

    mode: Literal["set_languages", "main_languages", "all_languages"]
    languages: Optional[list[str]]  # Required for set_languages mode


class SynthesisInput(TypedDict):
    """Input parameters for synthesis workflow."""

    topic: str
    research_questions: list[str]
    synthesis_brief: Optional[str]  # Describes desired synthesis angle
    quality: QualityTier
    language_code: Optional[str]  # ISO 639-1 code, default "en"
    multi_lang_config: Optional[MultiLangConfig]  # Optional multi-lang research


class WebResearchResult(TypedDict):
    """Result from a web_research worker."""

    iteration: int
    query: str
    final_report: str
    source_count: int
    langsmith_run_id: str
    status: str


class BookFindingResult(TypedDict):
    """Result from a book_finding worker."""

    iteration: int
    theme: str
    final_report: str
    processed_books: list[dict]
    zotero_keys: list[str]  # Zotero keys for synthesized books
    status: str


class GeneratedQuery(TypedDict):
    """A query generated for web research."""

    query: str
    rationale: str
    target_area: str  # What aspect of the topic this addresses


class GeneratedTheme(TypedDict):
    """A theme generated for book finding."""

    theme: str
    rationale: str
    book_angle: str  # "analogous", "inspiring", or "expressive"


class SynthesisSection(TypedDict):
    """A section of the synthesis document."""

    section_id: str
    title: str
    content: str
    citations: list[str]  # List of [@ZOTKEY] citations used
    quality_score: Optional[float]
    needs_revision: bool


class SelectedBook(TypedDict):
    """A book selected for deep integration in synthesis."""

    zotero_key: str
    title: str
    rationale: str  # Why this book was selected for synthesis


class SynthesisStructure(TypedDict):
    """Suggested structure for the synthesis document."""

    title: str
    sections: list[dict]  # [{section_id, title, description, key_sources}]
    introduction_guidance: str
    conclusion_guidance: str


# =============================================================================
# Main State
# =============================================================================


class SynthesisState(TypedDict):
    """Complete state for synthesis workflow."""

    # Input
    input: SynthesisInput
    quality_settings: dict[str, Any]

    # Phase 1: Literature Review
    lit_review_result: Optional[dict]  # Full result from academic_lit_review
    paper_corpus: dict[str, Any]  # DOI -> PaperMetadata
    paper_summaries: dict[str, Any]  # DOI -> PaperSummary
    zotero_keys: dict[str, str]  # DOI -> Zotero key

    # Phase 2: Supervision
    supervision_result: Optional[dict]  # Full result from enhance.supervision

    # Phase 3: Research Targets
    generated_queries: list[GeneratedQuery]
    generated_themes: list[GeneratedTheme]

    # Phase 3b: Parallel Research Results (aggregated via reducers)
    web_research_results: Annotated[list[WebResearchResult], add]
    book_finding_results: Annotated[list[BookFindingResult], add]

    # Phase 4: Synthesis
    synthesis_structure: Optional[SynthesisStructure]
    selected_books: list[SelectedBook]
    book_summaries_cache: dict[str, str]  # zotero_key -> 10x summary
    section_drafts: Annotated[list[SynthesisSection], add]

    # Phase 5: Editing
    editing_result: Optional[dict]

    # Output
    final_report: Optional[str]
    final_report_with_references: Optional[str]

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str
    status: Optional[str]  # "success", "partial", "failed"
    langsmith_run_id: Optional[str]
    errors: Annotated[list[dict], add]
