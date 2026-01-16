"""State definitions for the editing workflow."""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal, Optional

from typing_extensions import TypedDict


class EditingInput(TypedDict):
    """Input for the editing workflow."""

    document: str  # The document to edit (markdown)
    topic: str  # Topic/context for the document (helps with coherence)


class EditingState(TypedDict, total=False):
    """State for the structural editing workflow.

    Follows the standard workflow state pattern with reducers
    for accumulating results from parallel operations.
    """

    # === Input ===
    input: EditingInput
    quality_settings: dict[str, Any]  # Quality tier configuration

    # === Phase 1: Parse ===
    document_model: dict  # Serialized DocumentModel
    parse_complete: bool
    parse_warnings: list[str]

    # === Phase 2: Analyze ===
    structural_analysis: dict  # Serialized StructuralAnalysis
    analysis_complete: bool

    # === Phase 3: Plan ===
    edit_plan: dict  # Serialized EditPlan
    plan_complete: bool

    # === Phase 4: Execute ===
    # Accumulates results from parallel workers using add reducer
    completed_edits: Annotated[list[dict], add]
    execution_complete: bool

    # === Phase 5: Verify Structure ===
    updated_document_model: dict  # After structural edits
    structure_verification: dict
    needs_more_structure_work: bool

    # === Iteration Tracking (Structure) ===
    structure_iteration: int
    max_structure_iterations: int

    # === Citation Detection ===
    has_citations: bool  # Auto-detected from document
    citation_keys: list[str]  # All [@KEY] found

    # === Phase 6: Enhance (when has_citations=True) ===
    enhance_iteration: int  # Current iteration (0-based)
    max_enhance_iterations: int  # From quality settings
    section_enhancements: Annotated[list[dict], add]  # Results from parallel workers
    enhance_coherence_review: dict  # Holistic coherence after each iteration
    enhance_flagged_sections: list[str]  # Section IDs needing re-enhancement
    enhance_complete: bool

    # === Phase 7: Verify Facts (when has_citations=True) ===
    screened_sections: list[str]  # Section IDs that passed screening for fact-check
    screening_skipped: list[str]  # Section IDs skipped by screening
    fact_check_results: Annotated[list[dict], add]  # Per-section results
    reference_check_results: Annotated[list[dict], add]  # Citation validations
    citation_cache: dict[str, dict]  # Pre-validated citation existence cache
    pending_edits: Annotated[list[dict], add]  # Edits identified by verification (parallel)
    applied_edits: list[dict]  # Edits successfully applied
    skipped_edits: list[dict]  # Edits that couldn't be applied (logged)
    unresolved_items: Annotated[list[dict], add]  # Items logged at INFO (parallel workers)
    verify_complete: bool

    # === Phase 8: Polish ===
    polish_results: list[dict]
    polish_complete: bool

    # === Phase 9: Finalize ===
    final_document: str
    final_verification: dict

    # === Metadata (standard workflow fields) ===
    langsmith_run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: Optional[Literal["success", "partial", "failed"]]
    errors: Annotated[list[dict], add]
    changes_summary: str


def build_initial_state(
    document: str,
    topic: str,
    quality_settings: dict[str, Any],
    langsmith_run_id: str,
) -> EditingState:
    """Build initial state for editing workflow.

    Args:
        document: The document to edit
        topic: Topic/context for the document
        quality_settings: Quality tier settings
        langsmith_run_id: LangSmith run ID for tracing

    Returns:
        Initialized EditingState
    """
    max_structure_iterations = quality_settings.get("max_structure_iterations", 3)
    max_enhance_iterations = quality_settings.get("max_enhance_iterations", 3)

    return EditingState(
        input=EditingInput(document=document, topic=topic),
        quality_settings=quality_settings,
        # Phase tracking
        parse_complete=False,
        analysis_complete=False,
        plan_complete=False,
        execution_complete=False,
        enhance_complete=False,
        verify_complete=False,
        polish_complete=False,
        # Structure iteration control
        structure_iteration=0,
        max_structure_iterations=max_structure_iterations,
        needs_more_structure_work=False,
        # Citation detection (set during parse)
        has_citations=False,
        citation_keys=[],
        # Enhancement iteration control
        enhance_iteration=0,
        max_enhance_iterations=max_enhance_iterations,
        enhance_flagged_sections=[],
        # Accumulators (start empty, use add reducer)
        completed_edits=[],
        section_enhancements=[],
        fact_check_results=[],
        reference_check_results=[],
        errors=[],
        # Verification tracking
        pending_edits=[],
        applied_edits=[],
        skipped_edits=[],
        unresolved_items=[],
        # Metadata
        langsmith_run_id=langsmith_run_id,
        started_at=datetime.utcnow(),
        completed_at=None,
        status=None,
    )
