"""Wrapper nodes for enhancement supervision workflow.

These nodes bridge the EnhanceState to the standalone loop functions,
extracting parameters and updating state appropriately.
"""

import logging
from typing import Any

from langsmith import traceable
from langgraph.types import RunnableConfig

from workflows.enhance.supervision.loop1.graph import run_loop1_standalone
from workflows.enhance.supervision.loop2.graph import run_loop2_standalone
from workflows.enhance.supervision.types import EnhanceState

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="SupervisionLoop1Node")
async def run_loop1_node(state: EnhanceState, config: RunnableConfig) -> dict[str, Any]:
    """Run Loop 1 (theoretical depth) and update state.

    Extracts parameters from EnhanceState, calls run_loop1_standalone,
    and returns state updates.
    """
    input_data = state.get("input", {})
    current_review = state.get("current_review", "")
    quality_settings = state.get("quality_settings", {})
    max_iterations = state.get("max_iterations_per_loop", 3)
    paper_corpus = state.get("paper_corpus", {})
    checkpoint_callback = state.get("checkpoint_callback")
    incremental_state = state.get("incremental_state")

    logger.info(f"Running Loop 1 (theoretical depth) on {len(current_review)} char review")

    result = await run_loop1_standalone(
        review=current_review,
        topic=input_data.get("topic", ""),
        research_questions=input_data.get("research_questions", []),
        max_iterations=max_iterations,
        source_count=len(paper_corpus),
        quality_settings=quality_settings,
        config=config,
        checkpoint_callback=checkpoint_callback,
        incremental_state=incremental_state,
    )

    if result.errors:
        raise RuntimeError(
            f"Loop 1 finished with {len(result.errors)} error(s): "
            f"{result.errors[0].get('error_message') or result.errors[0]}"
        )

    logger.info(f"Loop 1 complete: {result.changes_summary}")

    return {
        "current_review": result.current_review,
        "review_loop1": result.current_review,
        "loop1_result": {
            "changes_summary": result.changes_summary,
            "issues_explored": result.issues_explored,
        },
        "loop_progress": [
            {
                "loop": "loop1",
                "changes_summary": result.changes_summary,
                "issues_explored": result.issues_explored,
            }
        ],
    }


@traceable(run_type="chain", name="SupervisionLoop2Node")
async def run_loop2_node(state: EnhanceState, config: RunnableConfig) -> dict[str, Any]:
    """Run Loop 2 (literature expansion) and update state.

    Extracts parameters from EnhanceState, calls run_loop2_standalone,
    and returns state updates including merged paper corpus.
    """
    input_data = state.get("input", {})
    current_review = state.get("current_review", "")
    quality_settings = state.get("quality_settings", {})
    max_iterations = state.get("max_iterations_per_loop", 3)
    paper_corpus = state.get("paper_corpus", {})
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})
    checkpoint_callback = state.get("checkpoint_callback")
    incremental_state = state.get("incremental_state")

    logger.info(
        f"Running Loop 2 (literature expansion) on {len(current_review)} char review "
        f"with {len(paper_corpus)} existing papers"
    )

    # Build LitReviewInput compatible dict for loop2
    lit_review_input = {
        "topic": input_data.get("topic", ""),
        "research_questions": input_data.get("research_questions", []),
        "quality": input_data.get("quality", "standard"),
        "date_range": None,
        "language_code": "en",
    }

    result = await run_loop2_standalone(
        review=current_review,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
        input_data=lit_review_input,
        quality_settings=quality_settings,
        max_iterations=max_iterations,
        config=config,
        checkpoint_callback=checkpoint_callback,
        incremental_state=incremental_state,
    )

    inner_errors = result.get("errors", []) or []
    if inner_errors:
        raise RuntimeError(
            f"Loop 2 finished with {len(inner_errors)} error(s): "
            f"{inner_errors[0].get('error_message') or inner_errors[0]}"
        )

    logger.info(
        f"Loop 2 complete: explored {len(result.get('explored_bases', []))} literature bases, "
        f"{len(result.get('paper_corpus', {}))} papers in corpus"
    )

    return {
        "current_review": result.get("current_review", current_review),
        "review_loop2": result.get("current_review", current_review),
        "paper_corpus": result.get("paper_corpus", paper_corpus),
        "paper_summaries": result.get("paper_summaries", paper_summaries),
        "zotero_keys": result.get("zotero_keys", zotero_keys),
        "loop2_result": {
            "explored_bases": result.get("explored_bases", []),
            "iteration": result.get("iteration", 0),
        },
        "loop_progress": [
            {
                "loop": "loop2",
                "explored_bases": result.get("explored_bases", []),
                "iteration": result.get("iteration", 0),
            }
        ],
    }


@traceable(run_type="chain", name="SupervisionFinalizeNode")
def finalize_node(state: EnhanceState) -> dict[str, Any]:
    """Finalize enhancement and prepare result.

    Sets final_review, completion_reason, and is_complete flag.
    """
    current_review = state.get("current_review", "")
    loop_progress = state.get("loop_progress", [])
    errors = state.get("errors", [])

    loops_run = [entry.get("loop", "unknown") for entry in loop_progress]

    if errors:
        completion_reason = f"Completed with {len(errors)} errors"
    elif not loops_run:
        completion_reason = "No loops executed"
    else:
        completion_reason = f"Successfully completed loops: {', '.join(loops_run)}"

    logger.info(f"Finalizing enhancement: {completion_reason}")

    return {
        "final_review": current_review,
        "completion_reason": completion_reason,
        "is_complete": True,
    }
