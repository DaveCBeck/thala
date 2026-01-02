"""
Main entry point for wrapped research workflow.

Provides the public wrapped_research() function that orchestrates
comprehensive research across web, academic, and book sources.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from workflows.wrapped.state import (
    WrappedResearchState,
    WrappedResearchInput,
    CheckpointPhase,
    QualityTier,
)
from workflows.wrapped.checkpointing import load_checkpoint, get_resume_phase, delete_checkpoint
from .construction import wrapped_research_graph

logger = logging.getLogger(__name__)


async def wrapped_research(
    query: str,
    quality: QualityTier = "standard",
    research_questions: Optional[list[str]] = None,
    date_range: Optional[tuple[int, int]] = None,
    resume_from: Optional[str] = None,
) -> dict[str, Any]:
    """Run comprehensive research across web, academic, and book sources.

    Orchestrates three research workflows:
    1. Web research (deep_research) - current trends, practical info
    2. Academic literature review (academic_lit_review) - empirical evidence
    3. Book finding (book_finding) - deep perspective, narrative synthesis

    Web and academic research run in parallel. After both complete, a thematic
    query is generated for book finding. Finally, all outputs are synthesized
    and saved to top_of_mind.

    Args:
        query: Research topic or question
        quality: Quality tier - "quick", "standard", or "comprehensive"
            - quick: Fast, focused research (~30 min total)
            - standard: Balanced depth (~2-4 hours)
            - comprehensive: Exhaustive research (hours to days)
        research_questions: Optional specific questions for academic review.
            If not provided, questions are auto-generated from the query.
        date_range: Optional (start_year, end_year) filter for academic papers
        resume_from: Optional LangSmith run ID to resume from checkpoint.
            Use this to continue a workflow that was interrupted.

    Returns:
        Dict containing:
        - web_result: Web research WorkflowResult
        - academic_result: Academic literature review WorkflowResult
        - book_result: Book recommendations WorkflowResult
        - combined_summary: LLM synthesis of all three sources
        - top_of_mind_ids: UUIDs of saved records {web, academic, books, combined}
        - started_at: Workflow start timestamp
        - completed_at: Workflow completion timestamp
        - langsmith_run_id: Run ID for tracing and resumption
        - errors: Any errors encountered during workflow

    Example:
        # Basic usage
        result = await wrapped_research(
            query="Impact of AI on creative work",
            quality="standard",
        )
        print(result["combined_summary"])

        # Resume interrupted workflow
        result = await wrapped_research(
            query="Impact of AI on creative work",
            resume_from="abc123-def456-...",
        )
    """
    run_id = str(uuid.uuid4())

    # Check for resume
    initial_state: Optional[WrappedResearchState] = None

    if resume_from:
        checkpoint_state = load_checkpoint(resume_from)
        if checkpoint_state:
            resume_phase = get_resume_phase(checkpoint_state)
            logger.info(f"Resuming from checkpoint: {resume_from}, phase: {resume_phase}")
            # Use the loaded state directly
            # Note: Full resume implementation would use conditional edges
            # to skip completed phases. For now, we restart from the checkpoint
            # but the checkpoint data is preserved.
            initial_state = checkpoint_state
            run_id = resume_from  # Keep the same run ID
        else:
            logger.warning(f"Checkpoint not found for run: {resume_from}, starting fresh")

    if initial_state is None:
        initial_state = WrappedResearchState(
            input=WrappedResearchInput(
                query=query,
                quality=quality,
                research_questions=research_questions,
                date_range=date_range,
            ),
            web_result=None,
            academic_result=None,
            book_result=None,
            book_theme=None,
            book_brief=None,
            combined_summary=None,
            top_of_mind_ids={},
            checkpoint_phase=CheckpointPhase(
                parallel_research=False,
                book_query_generated=False,
                book_finding=False,
                saved_to_top_of_mind=False,
            ),
            checkpoint_path=None,
            started_at=datetime.utcnow(),
            completed_at=None,
            current_phase="starting",
            langsmith_run_id=run_id,
            errors=[],
        )

    logger.info(
        f"Starting wrapped research: query='{query[:50]}...', "
        f"quality={quality}, run_id={run_id}"
    )

    try:
        result = await wrapped_research_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"wrapped_research:{query[:30]}",
            },
        )

        # Workflow completed successfully - clean up checkpoint
        delete_checkpoint(run_id)

        completed_at = datetime.utcnow()

        return {
            "web_result": result.get("web_result"),
            "academic_result": result.get("academic_result"),
            "book_result": result.get("book_result"),
            "combined_summary": result.get("combined_summary"),
            "top_of_mind_ids": result.get("top_of_mind_ids", {}),
            "started_at": initial_state["started_at"],
            "completed_at": completed_at,
            "langsmith_run_id": run_id,
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Wrapped research failed: {e}")
        # Don't delete checkpoint on failure - allow resume
        return {
            "web_result": None,
            "academic_result": None,
            "book_result": None,
            "combined_summary": None,
            "top_of_mind_ids": {},
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
            "langsmith_run_id": run_id,
            "errors": [{"phase": "orchestration", "error": str(e)}],
        }
