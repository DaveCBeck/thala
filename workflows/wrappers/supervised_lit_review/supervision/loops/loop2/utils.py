"""Loop 2 utility functions."""

import logging
from typing import Any

from langsmith import traceable

from workflows.shared.workflow_state_store import save_workflow_state

from .graph import Loop2Result, Loop2State, loop2_graph

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="Loop2_LiteratureExpansion")
async def run_loop2_standalone(
    review: str,
    topic: str,
    research_questions: list[str],
    quality_settings: dict[str, Any],
    config: dict | None = None,
) -> Loop2Result:
    """Run Loop 2 (literature base expansion) as a standalone operation.

    Args:
        review: The literature review text to analyze and improve
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality settings (max_stages used for iterations)
        config: Optional LangGraph config for tracing

    Returns:
        Loop2Result with improved review and metadata
    """
    # Derive max_iterations from quality settings (same pattern as orchestrator)
    max_iterations = quality_settings.get("max_stages", 3)

    initial_state = Loop2State(
        current_review=review,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        iteration=0,
        max_iterations=max_iterations,
        explored_bases=[],
        is_complete=False,
        decision=None,
        errors=[],
        iterations_failed=0,
        consecutive_failures=0,
        integration_failed=False,
        mini_review_failed=False,
    )

    logger.info(
        f"Starting Loop 2: max_iterations={max_iterations}, "
        f"review length={len(review)} chars"
    )

    if config:
        final_state = await loop2_graph.ainvoke(initial_state, config=config)
    else:
        final_state = await loop2_graph.ainvoke(initial_state)

    explored_bases = final_state.get("explored_bases", [])

    # Build changes summary
    if explored_bases:
        changes_summary = f"Expanded {len(explored_bases)} literature bases: {', '.join(explored_bases)}"
    else:
        changes_summary = "No literature bases identified for expansion"

    logger.info(f"Loop 2 complete: {len(explored_bases)} bases explored")

    # Save state for analysis (dev mode only)
    run_id = config.get("run_id", "unknown") if config else "unknown"
    save_workflow_state(
        workflow_name="supervision_loop2",
        run_id=str(run_id),
        state={
            "input": {
                "review_length": len(review),
                "topic": topic,
                "research_questions": research_questions,
                "max_iterations": max_iterations,
            },
            "output": {
                "review_length": len(final_state.get("current_review", review)),
                "explored_bases": explored_bases,
                "errors": final_state.get("errors", []),
            },
            "final_state": final_state,
        },
    )

    return Loop2Result(
        current_review=final_state.get("current_review", review),
        changes_summary=changes_summary,
        explored_bases=explored_bases,
    )
