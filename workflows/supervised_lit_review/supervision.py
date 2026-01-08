"""Supervision phase for supervised literature review workflow."""

import logging
from datetime import datetime
from typing import Any

from workflows.academic_lit_review.state import AcademicLitReviewState
from workflows.supervised_lit_review.supervision.orchestration import (
    run_supervision_configurable,
)

logger = logging.getLogger(__name__)


async def supervision_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 6: Multi-loop supervision to strengthen the literature review.

    Runs configurable supervision loops based on quality_settings.supervision_loops:
    - Loop 1: Theoretical depth expansion
    - Loop 2: Literature base expansion (missing perspectives)
    - Loop 3: Structure and cohesion editing
    - Loop 4: Section-level deep editing
    - Loop 4.5: Cohesion check (may return to Loop 3)
    - Loop 5: Fact and reference checking

    Args:
        state: Current workflow state containing final_review from synthesis

    Returns:
        State updates including:
            - final_review_v2: Improved review after supervision
            - supervision: Supervision state tracking
            - human_review_items: Items flagged for human review
            - paper_corpus: Updated with any new papers
            - paper_summaries: Updated with new summaries
    """
    final_review = state.get("final_review", "")
    quality_settings = state["quality_settings"]
    input_data = state["input"]

    # Get supervision loop config (default to "all")
    supervision_loops = quality_settings.get("supervision_loops", "all")

    logger.info(f"Starting supervision phase (loops={supervision_loops})")

    if not final_review:
        logger.warning("No final_review to supervise, skipping phase")
        return {
            "final_review_v2": None,
            "supervision": None,
            "current_phase": "complete",
            "current_status": "Supervision skipped (no review content)",
            "completed_at": datetime.utcnow(),
        }

    if supervision_loops == "none":
        logger.info("Supervision disabled via config, skipping phase")
        return {
            "final_review_v2": final_review,
            "supervision": None,
            "current_phase": "complete",
            "current_status": "Supervision skipped (disabled in config)",
            "completed_at": datetime.utcnow(),
        }

    paper_corpus = state.get("paper_corpus", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    zotero_keys = state.get("zotero_keys", {})

    # Calculate max iterations per loop based on quality tier
    max_stages = quality_settings.get("max_stages", 3)
    max_iterations_per_loop = max_stages  # Each loop gets this many iterations

    logger.info(
        f"Supervision starting with {len(final_review)} char review, "
        f"{len(paper_corpus)} papers in corpus, max_iterations_per_loop={max_iterations_per_loop}"
    )

    supervision_result = await run_supervision_configurable(
        review=final_review,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
        clusters=clusters,
        input_data=input_data,
        quality_settings=quality_settings,
        max_iterations_per_loop=max_iterations_per_loop,
        loops=supervision_loops,
    )

    final_review_v2 = supervision_result.get("final_review", final_review)
    loops_run = supervision_result.get("loops_run", [])
    human_review_items = supervision_result.get("human_review_items", [])
    completion_reason = supervision_result.get("completion_reason", "")

    # Collect added papers from results
    added_papers = supervision_result.get("paper_corpus", {})
    added_summaries = supervision_result.get("paper_summaries", {})

    # Filter to only new papers (not in original corpus)
    new_papers = {doi: p for doi, p in added_papers.items() if doi not in paper_corpus}
    new_summaries = {doi: s for doi, s in added_summaries.items() if doi not in paper_summaries}

    logger.info(
        f"Supervision complete: loops={loops_run}, "
        f"{len(new_papers)} new papers, {len(human_review_items)} items for review. "
        f"Reason: {completion_reason}"
    )

    return {
        "final_review_v2": final_review_v2,
        "supervision": {
            "loops_run": loops_run,
            "completion_reason": completion_reason,
            "loop_progress": supervision_result.get("loop_progress"),
        },
        "human_review_items": human_review_items,
        "paper_corpus": new_papers,  # Will be merged by reducer
        "paper_summaries": new_summaries,  # Will be merged by reducer
        "current_phase": "complete",
        "current_status": f"Supervision complete: {len(loops_run)} loops ({completion_reason})",
        "completed_at": datetime.utcnow(),
    }
