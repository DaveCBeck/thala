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

    V2 Structure Phase: Uses simplified 3-phase approach (analyze → rewrite → reassemble)
    before transitioning to V1 Enhancement and Polish phases via bridge node.
    """

    # === Input ===
    input: EditingInput
    quality_settings: dict[str, Any]  # Quality tier configuration

    # === V2 Structure Phase 1: Analyze ===
    sections: list[dict]  # V2 TopLevelSection list
    edit_instructions: list[dict]  # V2 EditInstruction list
    analysis_complete: bool

    # === V2 Structure Phase 2: Rewrite ===
    # Accumulates results from parallel rewriters using add reducer
    rewritten_sections: Annotated[list[dict], add]  # V2 RewrittenSection results
    rewriting_complete: bool

    # === V2 Structure Phase 3: Reassemble ===
    final_document: str  # V2 output (markdown) before bridge
    verification: dict  # V2 coherence verification

    # === Bridge (V2 → V1) ===
    updated_document_model: dict  # DocumentModel for Enhancement phase

    # === Citation Detection ===
    has_citations: bool  # Auto-detected from document
    citation_keys: list[str]  # All [@KEY] found

    # === Enhancement Phase (when has_citations=True) ===
    enhance_iteration: int  # Current iteration (0-based)
    max_enhance_iterations: int  # From quality settings
    section_enhancements: Annotated[list[dict], add]  # Results from parallel workers
    enhance_coherence_review: dict  # Holistic coherence after each iteration
    enhance_flagged_sections: list[str]  # Section IDs needing re-enhancement
    enhance_complete: bool

    # === Polish Phase ===
    polish_results: list[dict]
    polish_complete: bool

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
    max_enhance_iterations = quality_settings.get("max_enhance_iterations", 3)

    return EditingState(
        input=EditingInput(document=document, topic=topic),
        quality_settings=quality_settings,
        # V2 Structure Phase tracking
        analysis_complete=False,
        rewriting_complete=False,
        # V2 Structure Phase accumulators
        sections=[],
        edit_instructions=[],
        rewritten_sections=[],
        # Citation detection (set by bridge node)
        has_citations=False,
        citation_keys=[],
        # Enhancement iteration control
        enhance_iteration=0,
        max_enhance_iterations=max_enhance_iterations,
        enhance_flagged_sections=[],
        enhance_complete=False,
        # Polish tracking
        polish_complete=False,
        # Accumulators (start empty, use add reducer)
        section_enhancements=[],
        errors=[],
        # Metadata
        langsmith_run_id=langsmith_run_id,
        started_at=datetime.utcnow(),
        completed_at=None,
        status=None,
    )
