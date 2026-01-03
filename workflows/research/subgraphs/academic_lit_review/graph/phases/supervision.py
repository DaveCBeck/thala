"""Supervision phase node for academic literature review workflow."""

import logging
from datetime import datetime
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import AcademicLitReviewState
from workflows.research.subgraphs.academic_lit_review.supervision.graph import (
    run_supervision,
)

logger = logging.getLogger(__name__)


async def supervision_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 6: Iterative supervision to strengthen theoretical depth.

    Reviews final_review for under-explored theories, runs focused expansions,
    and integrates findings. Loops until max_iterations or pass-through.

    Args:
        state: Current workflow state containing final_review from synthesis

    Returns:
        State updates including:
            - final_review_v2: Improved review after supervision
            - supervision: Supervision state tracking
            - supervision_expansions: Record of expansions performed
            - paper_corpus: Updated with any new papers
            - paper_summaries: Updated with new summaries
    """
    final_review = state.get("final_review", "")
    quality_settings = state["quality_settings"]
    input_data = state["input"]

    logger.info("Starting supervision phase")

    if not final_review:
        logger.warning("No final_review to supervise, skipping phase")
        return {
            "final_review_v2": None,
            "supervision": None,
            "current_phase": "complete",
            "current_status": "Supervision skipped (no review content)",
            "completed_at": datetime.utcnow(),
        }

    paper_corpus = state.get("paper_corpus", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    zotero_keys = state.get("zotero_keys", {})

    logger.info(
        f"Supervision starting with {len(final_review)} char review, "
        f"{len(paper_corpus)} papers in corpus"
    )

    supervision_result = await run_supervision(
        final_review=final_review,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        clusters=clusters,
        quality_settings=quality_settings,
        input_data=input_data,
        zotero_keys=zotero_keys,
    )

    final_review_v2 = supervision_result.get("final_review_v2", final_review)
    supervision_state = supervision_result.get("supervision_state")
    expansions = supervision_result.get("expansions", [])
    iterations = supervision_result.get("iterations", 0)
    added_papers = supervision_result.get("added_papers", {})
    added_summaries = supervision_result.get("added_summaries", {})
    completion_reason = supervision_result.get("completion_reason", "")

    logger.info(
        f"Supervision complete: {iterations} iterations, "
        f"{len(expansions)} expansions, {len(added_papers)} new papers. "
        f"Reason: {completion_reason}"
    )

    return {
        "final_review_v2": final_review_v2,
        "supervision": supervision_state,
        "supervision_expansions": expansions,
        "paper_corpus": added_papers,  # Will be merged by reducer
        "paper_summaries": added_summaries,  # Will be merged by reducer
        "current_phase": "complete",
        "current_status": f"Supervision complete: {iterations} iterations ({completion_reason})",
        "completed_at": datetime.utcnow(),
    }
