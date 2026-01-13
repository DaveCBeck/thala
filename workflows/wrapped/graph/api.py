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
    QualityTier,
)
from workflows.shared.workflow_state_store import save_workflow_state
from .construction import wrapped_research_graph

logger = logging.getLogger(__name__)


async def wrapped_research(
    query: str,
    quality: QualityTier = "standard",
    research_questions: Optional[list[str]] = None,
    date_range: Optional[tuple[int, int]] = None,
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

    Returns:
        Dict containing standardized fields:
        - final_report: Combined synthesis of all three sources
        - status: "success", "partial", or "failed"
        - langsmith_run_id: Run ID for tracing and state retrieval
        - errors: Any errors encountered during workflow
        - source_count: Total sources across all three workflows
        - started_at: Workflow start timestamp
        - completed_at: Workflow completion timestamp

        For detailed results (web_result, academic_result, book_result, top_of_mind_ids),
        use load_workflow_state("wrapped_research", langsmith_run_id).

    Example:
        result = await wrapped_research(
            query="Impact of AI on creative work",
            quality="standard",
        )
        print(result["combined_summary"])
    """
    run_id = str(uuid.uuid4())

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
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="starting",
        langsmith_run_id=run_id,
        errors=[],
    )

    logger.info(
        f"Starting wrapped research for '{query[:50]}...' with quality={quality}"
    )
    logger.debug(f"Run ID: {run_id}")

    try:
        result = await wrapped_research_graph.ainvoke(
            initial_state,
            config={
                "run_id": run_id,
                "run_name": f"wrapped_research:{query[:30]}",
            },
        )

        completed_at = datetime.utcnow()
        combined_summary = result.get("combined_summary")
        errors = result.get("errors", [])

        # Determine standardized status
        if combined_summary and not errors:
            status = "success"
        elif combined_summary and errors:
            status = "partial"
        else:
            status = "failed"

        # Save full state for downstream workflows (in dev/test mode)
        save_workflow_state(
            workflow_name="wrapped_research",
            run_id=run_id,
            state={
                "input": dict(initial_state.get("input", {})),
                "web_result": result.get("web_result"),
                "academic_result": result.get("academic_result"),
                "book_result": result.get("book_result"),
                "book_theme": result.get("book_theme"),
                "book_brief": result.get("book_brief"),
                "combined_summary": combined_summary,
                "top_of_mind_ids": result.get("top_of_mind_ids", {}),
                "started_at": initial_state["started_at"],
                "completed_at": completed_at,
            },
        )

        # Count sources from all three workflows
        source_count = 0
        if result.get("web_result"):
            source_count += result["web_result"].get("source_count", 0)
        if result.get("academic_result"):
            source_count += result["academic_result"].get("source_count", 0)
        if result.get("book_result"):
            source_count += result["book_result"].get("source_count", 0)

        logger.info(f"Wrapped research complete: status={status}, sources={source_count}")

        return {
            "final_report": combined_summary,
            "status": status,
            "langsmith_run_id": run_id,
            "errors": errors,
            "source_count": source_count,
            "started_at": initial_state["started_at"],
            "completed_at": completed_at,
        }

    except Exception as e:
        logger.error(f"Wrapped research failed: {e}")
        return {
            "final_report": None,
            "status": "failed",
            "langsmith_run_id": run_id,
            "errors": [{"phase": "orchestration", "error": str(e)}],
            "source_count": 0,
            "started_at": initial_state["started_at"],
            "completed_at": datetime.utcnow(),
        }
