"""Orchestration graph for multi-loop supervision."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from .types import OrchestrationState
from ..graph import run_supervision
from ..loops.loop2_literature import run_loop2_standalone
from ..loops.loop3_structure import run_loop3_standalone
from ..loops.loop4_editing import run_loop4_standalone
from ..loops.loop4_5_cohesion import check_cohesion
from ..loops.loop5_factcheck import run_loop5_standalone
from ..utils import document_revision
from workflows.research.subgraphs.academic_lit_review.state import (
    MultiLoopProgress,
    LoopCheckpoint,
    RevisionRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Nodes
# =============================================================================


async def run_loop1_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 1: Theoretical depth expansion via existing supervision."""
    logger.info("Running Loop 1: Theoretical depth expansion")

    before_text = state["current_review"]

    # Determine max iterations from shared budget
    loop_progress = state["loop_progress"]
    remaining_budget = loop_progress["shared_iteration_budget"]
    max_iterations = min(3, remaining_budget)

    if max_iterations <= 0:
        logger.warning("No iteration budget remaining for Loop 1, skipping")
        return {
            "loop1_result": {"iterations": 0, "completion_reason": "No budget"},
            "loop_progress": loop_progress,
        }

    result = await run_supervision(
        final_review=state["current_review"],
        paper_corpus=state["paper_corpus"],
        paper_summaries=state["paper_summaries"],
        clusters=state["clusters"],
        quality_settings=state["quality_settings"],
        input_data=state["input"],
        zotero_keys=state["zotero_keys"],
    )

    after_text = result.get("final_review_v2", state["current_review"])
    iterations_used = result.get("iterations", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_1"] = iterations_used
    loop_progress["shared_iteration_budget"] -= iterations_used
    loop_progress["current_loop"] = 2

    # Document revision
    if before_text != after_text:
        revision = await document_revision(
            loop_number=1,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    # Checkpoint
    checkpoint = LoopCheckpoint(
        loop_number=1,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",  # Add timestamp in production
    )
    loop_progress["checkpoints"].append(checkpoint)

    logger.info(f"Loop 1 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "loop1_result": result,
        "loop_progress": loop_progress,
        "paper_corpus": result.get("added_papers", {}),
        "paper_summaries": result.get("added_summaries", {}),
    }


async def run_loop2_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 2: Literature base expansion."""
    logger.info("Running Loop 2: Literature base expansion")

    before_text = state["current_review"]

    loop_progress = state["loop_progress"]
    remaining_budget = loop_progress["shared_iteration_budget"]
    max_iterations = min(3, remaining_budget)

    if max_iterations <= 0:
        logger.warning("No iteration budget remaining for Loop 2, skipping")
        return {
            "loop2_result": {"iterations": 0, "completion_reason": "No budget"},
            "loop_progress": loop_progress,
        }

    result = await run_loop2_standalone(
        review=state["current_review"],
        paper_corpus=state["paper_corpus"],
        paper_summaries=state["paper_summaries"],
        zotero_keys=state["zotero_keys"],
        input_data=state["input"],
        quality_settings=state["quality_settings"],
        max_iterations=max_iterations,
    )

    after_text = result.get("current_review", state["current_review"])
    iterations_used = result.get("iteration", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_2"] = iterations_used
    loop_progress["shared_iteration_budget"] -= iterations_used
    loop_progress["current_loop"] = 3

    # Document revision
    if before_text != after_text:
        revision = await document_revision(
            loop_number=2,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    # Checkpoint
    checkpoint = LoopCheckpoint(
        loop_number=2,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

    logger.info(f"Loop 2 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "loop2_result": result,
        "loop_progress": loop_progress,
        "paper_corpus": result.get("paper_corpus", state["paper_corpus"]),
        "paper_summaries": result.get("paper_summaries", state["paper_summaries"]),
        "zotero_keys": result.get("zotero_keys", state["zotero_keys"]),
    }


async def run_loop3_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 3: Structure and cohesion."""
    logger.info("Running Loop 3: Structure and cohesion")

    before_text = state["current_review"]

    loop_progress = state["loop_progress"]
    remaining_budget = loop_progress["shared_iteration_budget"]
    max_iterations = min(3, remaining_budget)

    if max_iterations <= 0:
        logger.warning("No iteration budget remaining for Loop 3, skipping")
        return {
            "loop3_result": {"iterations": 0, "completion_reason": "No budget"},
            "loop_progress": loop_progress,
        }

    result = await run_loop3_standalone(
        review=state["current_review"],
        input_data=state["input"],
        max_iterations=max_iterations,
    )

    after_text = result.get("current_review", state["current_review"])
    iterations_used = result.get("iteration", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_3"] = iterations_used
    loop_progress["shared_iteration_budget"] -= iterations_used
    loop_progress["current_loop"] = 4

    # Document revision
    if before_text != after_text:
        revision = await document_revision(
            loop_number=3,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    # Checkpoint
    checkpoint = LoopCheckpoint(
        loop_number=3,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

    logger.info(f"Loop 3 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "loop3_result": result,
        "loop_progress": loop_progress,
    }


async def run_loop4_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 4: Section-level deep editing."""
    logger.info("Running Loop 4: Section-level deep editing")

    before_text = state["current_review"]

    loop_progress = state["loop_progress"]
    remaining_budget = loop_progress["shared_iteration_budget"]
    max_iterations = min(3, remaining_budget)

    if max_iterations <= 0:
        logger.warning("No iteration budget remaining for Loop 4, skipping")
        return {
            "loop4_result": {"iterations": 0, "completion_reason": "No budget"},
            "loop_progress": loop_progress,
        }

    result = await run_loop4_standalone(
        review=state["current_review"],
        paper_summaries=state["paper_summaries"],
        input_data=state["input"],
        max_iterations=max_iterations,
    )

    after_text = result.get("edited_review", state["current_review"])
    iterations_used = result.get("iterations", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_4"] = iterations_used
    loop_progress["shared_iteration_budget"] -= iterations_used
    loop_progress["current_loop"] = 4.5  # Move to Loop 4.5

    # Document revision
    if before_text != after_text:
        revision = await document_revision(
            loop_number=4,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    # Checkpoint
    checkpoint = LoopCheckpoint(
        loop_number=4,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

    logger.info(f"Loop 4 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "loop4_result": result,
        "loop_progress": loop_progress,
    }


async def run_loop4_5_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 4.5: Cohesion check."""
    logger.info("Running Loop 4.5: Cohesion check")

    result = await check_cohesion(state["current_review"])

    logger.info(
        f"Loop 4.5 complete: needs_restructuring={result.needs_restructuring}"
    )

    return {
        "loop4_5_result": {
            "needs_restructuring": result.needs_restructuring,
            "reasoning": result.reasoning,
        },
    }


async def run_loop5_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 5: Fact and reference checking."""
    logger.info("Running Loop 5: Fact and reference checking")

    before_text = state["current_review"]

    result = await run_loop5_standalone(
        review=state["current_review"],
        paper_summaries=state["paper_summaries"],
        zotero_keys=state["zotero_keys"],
        max_iterations=1,  # Loop 5 typically runs once
    )

    after_text = result.get("current_review", state["current_review"])

    loop_progress = state["loop_progress"]
    loop_progress["loop_iterations"]["loop_5"] = 1
    loop_progress["current_loop"] = 6  # Complete

    # Document revision
    if before_text != after_text:
        revision = await document_revision(
            loop_number=5,
            iteration=1,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    # Checkpoint
    checkpoint = LoopCheckpoint(
        loop_number=5,
        iteration=1,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

    # Collect human review items
    human_review_items = result.get("human_review_items", [])

    logger.info(
        f"Loop 5 complete: {len(human_review_items)} items flagged for human review"
    )

    return {
        "current_review": after_text,
        "loop5_result": result,
        "loop_progress": loop_progress,
        "human_review_items": human_review_items,
    }


def finalize_node(state: OrchestrationState) -> dict[str, Any]:
    """Finalize orchestration and prepare final outputs."""
    logger.info("Finalizing multi-loop orchestration")

    loop_progress = state["loop_progress"]
    total_iterations = sum(loop_progress["loop_iterations"].values())

    completion_reason = (
        f"All loops complete. Total iterations: {total_iterations}. "
        f"Budget remaining: {loop_progress['shared_iteration_budget']}"
    )

    return {
        "final_review": state["current_review"],
        "is_complete": True,
        "completion_reason": completion_reason,
    }


# =============================================================================
# Routing
# =============================================================================


def increment_loop3_repeat_node(state: OrchestrationState) -> dict[str, Any]:
    """Increment loop3_repeat_count when returning from Loop 4.5 to Loop 3."""
    current_count = state.get("loop3_repeat_count", 0)
    logger.info(f"Incrementing loop3_repeat_count from {current_count} to {current_count + 1}")
    return {"loop3_repeat_count": current_count + 1}


def route_after_loop4_5(state: OrchestrationState) -> str:
    """Route after Loop 4.5 cohesion check.

    If needs_restructuring AND loop3_repeat_count < 1: return to Loop 3
    Otherwise: proceed to Loop 5
    """
    loop4_5_result = state.get("loop4_5_result", {})
    needs_restructuring = loop4_5_result.get("needs_restructuring", False)
    repeat_count = state.get("loop3_repeat_count", 0)

    if needs_restructuring and repeat_count < 1:
        logger.info("Cohesion check: returning to Loop 3 for restructuring")
        return "increment_and_loop3"
    else:
        if needs_restructuring:
            logger.warning(
                "Cohesion check flagged restructuring but max repeats reached, "
                "proceeding to Loop 5"
            )
        else:
            logger.info("Cohesion check passed, proceeding to Loop 5")
        return "loop5"


# =============================================================================
# Graph Construction
# =============================================================================


def create_orchestration_graph() -> StateGraph:
    """Create the multi-loop orchestration graph.

    Flow:
        START → loop1 → loop2 → loop3 → loop4 → loop4_5 → route
            → (needs_restructuring AND repeat_count < 1) → increment_and_loop3 → loop3
            → (approved OR max_repeat) → loop5 → finalize → END
    """
    builder = StateGraph(OrchestrationState)

    # Add nodes
    builder.add_node("loop1", run_loop1_node)
    builder.add_node("loop2", run_loop2_node)
    builder.add_node("loop3", run_loop3_node)
    builder.add_node("loop4", run_loop4_node)
    builder.add_node("loop4_5", run_loop4_5_node)
    builder.add_node("increment_and_loop3", increment_loop3_repeat_node)
    builder.add_node("loop5", run_loop5_node)
    builder.add_node("finalize", finalize_node)

    # Build flow
    builder.add_edge(START, "loop1")
    builder.add_edge("loop1", "loop2")
    builder.add_edge("loop2", "loop3")
    builder.add_edge("loop3", "loop4")
    builder.add_edge("loop4", "loop4_5")

    # Route after Loop 4.5
    builder.add_conditional_edges(
        "loop4_5",
        route_after_loop4_5,
        {
            "increment_and_loop3": "increment_and_loop3",  # Increment counter first
            "loop5": "loop5",  # Proceed to fact check
        },
    )

    # After incrementing, return to loop3
    builder.add_edge("increment_and_loop3", "loop3")

    builder.add_edge("loop5", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# =============================================================================
# Standalone API
# =============================================================================


async def run_supervision_orchestration(
    review: str,
    paper_corpus: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    clusters: list,
    input_data: dict,
    quality_settings: dict,
    max_shared_iterations: int = 10,
) -> dict:
    """Run full supervision orchestration through all loops.

    Args:
        review: Initial literature review text
        paper_corpus: DOI -> PaperMetadata mapping
        paper_summaries: DOI -> PaperSummary mapping
        zotero_keys: DOI -> Zotero key mapping
        clusters: Thematic clusters from main workflow
        input_data: LitReviewInput with topic and research questions
        quality_settings: Quality tier settings
        max_shared_iterations: Total iteration budget across all loops

    Returns:
        Dictionary containing:
            - final_review: Final review after all loops
            - loop_progress: Multi-loop progress tracking
            - human_review_items: Items flagged for human review
            - completion_reason: Why orchestration completed
            - loop1_result through loop5_result: Individual loop outputs
    """
    # Initialize loop progress
    loop_progress = MultiLoopProgress(
        current_loop=1,
        loop_iterations={
            "loop_1": 0,
            "loop_2": 0,
            "loop_3": 0,
            "loop_4": 0,
            "loop_5": 0,
        },
        shared_iteration_budget=max_shared_iterations,
        max_shared_iterations=max_shared_iterations,
        checkpoints=[],
        revision_history=[],
        loop3_repeat_count=0,
    )

    initial_state = OrchestrationState(
        current_review=review,
        final_review=None,
        input=input_data,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        clusters=clusters,
        quality_settings=quality_settings,
        zotero_keys=zotero_keys,
        loop_progress=loop_progress,
        loop3_repeat_count=0,
        loop1_result=None,
        loop2_result=None,
        loop3_result=None,
        loop4_result=None,
        loop4_5_result=None,
        loop5_result=None,
        human_review_items=[],
        completion_reason="",
        is_complete=False,
    )

    graph = create_orchestration_graph()
    final_state = await graph.ainvoke(initial_state)

    logger.info(
        f"Orchestration complete: {final_state.get('completion_reason', 'Unknown')}"
    )

    return {
        "final_review": final_state.get("final_review", review),
        "loop_progress": final_state.get("loop_progress", loop_progress),
        "human_review_items": final_state.get("human_review_items", []),
        "completion_reason": final_state.get("completion_reason", ""),
        "loop1_result": final_state.get("loop1_result"),
        "loop2_result": final_state.get("loop2_result"),
        "loop3_result": final_state.get("loop3_result"),
        "loop4_result": final_state.get("loop4_result"),
        "loop4_5_result": final_state.get("loop4_5_result"),
        "loop5_result": final_state.get("loop5_result"),
    }


# Valid supervision_loops config values
SUPERVISION_LOOP_CONFIGS = {
    "none": 0,   # Skip supervision entirely
    "one": 1,    # Loop 1 only (theoretical depth)
    "two": 2,    # Loops 1-2 (depth + literature bases)
    "three": 3,  # Loops 1-3 (depth + literature + structure)
    "four": 4,   # Loops 1-4 (depth + literature + structure + editing)
    "all": 5,    # All loops including fact-checking
    "five": 5,   # Alias for "all"
}


async def run_supervision_configurable(
    review: str,
    paper_corpus: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    clusters: list,
    input_data: dict,
    quality_settings: dict,
    max_shared_iterations: int = 10,
    loops: str = "all",
) -> dict:
    """Run configurable supervision with specified number of loops.

    Args:
        review: Initial literature review text
        paper_corpus: DOI -> PaperMetadata mapping
        paper_summaries: DOI -> PaperSummary mapping
        zotero_keys: DOI -> Zotero key mapping
        clusters: Thematic clusters from main workflow
        input_data: LitReviewInput with topic and research questions
        quality_settings: Quality tier settings
        max_shared_iterations: Total iteration budget across all loops
        loops: Which loops to run - "none", "one", "two", "three", "four", "all"

    Returns:
        Dictionary containing:
            - final_review: Final review after supervision
            - loop_progress: Multi-loop progress tracking
            - human_review_items: Items flagged for human review
            - completion_reason: Why supervision completed
            - loops_run: Which loops were executed
    """
    loop_count = SUPERVISION_LOOP_CONFIGS.get(loops.lower(), 5)

    if loop_count == 0:
        logger.info("Supervision skipped (loops='none')")
        return {
            "final_review": review,
            "loop_progress": None,
            "human_review_items": [],
            "completion_reason": "Supervision disabled",
            "loops_run": [],
        }

    # For full orchestration, use the graph-based approach
    if loop_count >= 5:
        logger.info("Running full multi-loop supervision (all loops)")
        result = await run_supervision_orchestration(
            review=review,
            paper_corpus=paper_corpus,
            paper_summaries=paper_summaries,
            zotero_keys=zotero_keys,
            clusters=clusters,
            input_data=input_data,
            quality_settings=quality_settings,
            max_shared_iterations=max_shared_iterations,
        )
        result["loops_run"] = ["loop1", "loop2", "loop3", "loop4", "loop4_5", "loop5"]
        return result

    # For partial orchestration, run loops sequentially
    logger.info(f"Running partial supervision (loops='{loops}', count={loop_count})")

    current_review = review
    current_corpus = paper_corpus.copy()
    current_summaries = paper_summaries.copy()
    current_zotero = zotero_keys.copy()
    loops_run = []
    all_results = {}
    human_review_items = []

    # Calculate per-loop iteration budget
    budget_per_loop = max(1, max_shared_iterations // loop_count)

    # Loop 1: Theoretical Depth (always runs if loop_count >= 1)
    if loop_count >= 1:
        logger.info("Running Loop 1: Theoretical depth expansion")
        loops_run.append("loop1")

        loop1_result = await run_supervision(
            final_review=current_review,
            paper_corpus=current_corpus,
            paper_summaries=current_summaries,
            clusters=clusters,
            quality_settings=quality_settings,
            input_data=input_data,
            zotero_keys=current_zotero,
        )

        current_review = loop1_result.get("final_review_v2", current_review)
        all_results["loop1_result"] = loop1_result

        # Merge any added papers
        added_papers = loop1_result.get("added_papers", {})
        added_summaries = loop1_result.get("added_summaries", {})
        current_corpus.update(added_papers)
        current_summaries.update(added_summaries)

    # Loop 2: Literature Base Expansion
    if loop_count >= 2:
        logger.info("Running Loop 2: Literature base expansion")
        loops_run.append("loop2")

        loop2_result = await run_loop2_standalone(
            review=current_review,
            paper_corpus=current_corpus,
            paper_summaries=current_summaries,
            zotero_keys=current_zotero,
            input_data=input_data,
            quality_settings=quality_settings,
            max_iterations=budget_per_loop,
        )

        current_review = loop2_result.get("current_review", current_review)
        current_corpus = loop2_result.get("paper_corpus", current_corpus)
        current_summaries = loop2_result.get("paper_summaries", current_summaries)
        current_zotero = loop2_result.get("zotero_keys", current_zotero)
        all_results["loop2_result"] = loop2_result

    # Loop 3: Structure and Cohesion
    if loop_count >= 3:
        logger.info("Running Loop 3: Structure and cohesion")
        loops_run.append("loop3")

        loop3_result = await run_loop3_standalone(
            review=current_review,
            input_data=input_data,
            max_iterations=budget_per_loop,
        )

        current_review = loop3_result.get("current_review", current_review)
        all_results["loop3_result"] = loop3_result

    # Loop 4 + 4.5: Section Editing + Cohesion Check
    if loop_count >= 4:
        logger.info("Running Loop 4: Section-level deep editing")
        loops_run.append("loop4")

        loop4_result = await run_loop4_standalone(
            review=current_review,
            paper_summaries=current_summaries,
            input_data=input_data,
            max_iterations=budget_per_loop,
        )

        current_review = loop4_result.get("edited_review", current_review)
        all_results["loop4_result"] = loop4_result

        # Run Loop 4.5 cohesion check
        logger.info("Running Loop 4.5: Cohesion check")
        loops_run.append("loop4_5")

        cohesion_result = await check_cohesion(current_review)
        all_results["loop4_5_result"] = {
            "needs_restructuring": cohesion_result.needs_restructuring,
            "reasoning": cohesion_result.reasoning,
        }

        # If needs restructuring, run Loop 3 again (once)
        if cohesion_result.needs_restructuring:
            logger.info("Cohesion check flagged issues - running Loop 3 again")
            loop3_repeat = await run_loop3_standalone(
                review=current_review,
                input_data=input_data,
                max_iterations=2,  # Limited iterations for repeat
            )
            current_review = loop3_repeat.get("current_review", current_review)

    completion_reason = f"Completed {loop_count} supervision loop(s)"
    logger.info(f"Partial supervision complete: {completion_reason}")

    return {
        "final_review": current_review,
        "loop_progress": None,  # Simplified tracking for partial runs
        "human_review_items": human_review_items,
        "completion_reason": completion_reason,
        "loops_run": loops_run,
        "paper_corpus": current_corpus,
        "paper_summaries": current_summaries,
        "zotero_keys": current_zotero,
        **all_results,
    }
