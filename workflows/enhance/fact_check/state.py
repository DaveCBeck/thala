"""State definitions for the fact-check workflow."""

from datetime import datetime, timezone
from operator import add
from typing import Annotated, Any, Literal, Optional

from typing_extensions import TypedDict


class FactCheckInput(TypedDict):
    """Input for the fact-check workflow."""

    document: Optional[str]  # The document to check (markdown) - optional if document_model provided
    document_model: Optional[dict]  # Pre-parsed document model (avoids re-parsing)
    topic: str  # Topic/context for the document


class FactCheckState(TypedDict, total=False):
    """State for the fact-check verification workflow.

    Follows the standard workflow state pattern with reducers
    for accumulating results from parallel operations.
    """

    # === Input ===
    input: FactCheckInput
    quality_settings: dict[str, Any]  # Quality tier configuration

    # === Phase 1: Parse ===
    document_model: dict  # Serialized DocumentModel
    parse_complete: bool
    parse_warnings: list[str]

    # === Phase 2: Citation Detection ===
    has_citations: bool  # Auto-detected from document
    citation_keys: list[str]  # All [@KEY] found

    # === Phase 3: Screening ===
    screened_sections: list[str]  # Section IDs that passed screening for fact-check
    screening_skipped: list[str]  # Section IDs skipped by screening

    # === Phase 4: Fact-check ===
    fact_check_results: Annotated[list[dict], add]  # Per-section results

    # === Phase 5: Reference-check ===
    citation_cache: dict[str, dict]  # Pre-validated citation existence cache
    reference_check_results: Annotated[list[dict], add]  # Citation validations

    # === Phase 6: Apply Edits ===
    pending_edits: Annotated[list[dict], add]  # Edits identified by verification (parallel)
    applied_edits: list[dict]  # Edits successfully applied
    skipped_edits: list[dict]  # Edits that couldn't be applied (logged)
    unresolved_items: Annotated[list[dict], add]  # Items logged at INFO (parallel workers)
    verify_complete: bool

    # === Phase 7: Finalize ===
    updated_document_model: dict  # After edits applied
    final_document: str

    # === Metadata (standard workflow fields) ===
    langsmith_run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    status: Optional[Literal["success", "partial", "failed", "skipped"]]
    errors: Annotated[list[dict], add]
    changes_summary: str


def build_initial_state(
    document: Optional[str],
    document_model: Optional[dict],
    topic: str,
    has_citations: Optional[bool],
    citation_keys: Optional[list[str]],
    quality_settings: dict[str, Any],
    langsmith_run_id: str,
) -> FactCheckState:
    """Build initial state for fact-check workflow.

    Args:
        document: The document to check (can be None if document_model provided)
        document_model: Pre-parsed document model (avoids re-parsing)
        topic: Topic/context for the document
        has_citations: Pre-detected citation flag (from editing workflow)
        citation_keys: Pre-detected citation keys (from editing workflow)
        quality_settings: Quality tier settings
        langsmith_run_id: LangSmith run ID for tracing

    Returns:
        Initialized FactCheckState
    """
    return FactCheckState(
        input=FactCheckInput(
            document=document,
            document_model=document_model,
            topic=topic,
        ),
        quality_settings=quality_settings,
        # Phase tracking
        parse_complete=False,
        verify_complete=False,
        # Citation detection (may be pre-set from editing)
        has_citations=has_citations if has_citations is not None else False,
        citation_keys=citation_keys or [],
        # Screening
        screened_sections=[],
        screening_skipped=[],
        # Accumulators (start empty, use add reducer)
        fact_check_results=[],
        reference_check_results=[],
        pending_edits=[],
        unresolved_items=[],
        errors=[],
        # Edit tracking
        applied_edits=[],
        skipped_edits=[],
        # Metadata
        langsmith_run_id=langsmith_run_id,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        status=None,
    )
