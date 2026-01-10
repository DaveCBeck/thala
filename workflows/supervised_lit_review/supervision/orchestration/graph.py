"""Orchestration graph for multi-loop supervision."""

import logging
import uuid
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
from ..utils.citation_validation import (
    validate_corpus_zotero_keys,
    CITATION_SOURCE_INITIAL,
    CITATION_SOURCE_LOOP1,
    CITATION_SOURCE_LOOP2,
    CITATION_SOURCE_LOOP4,
    CITATION_SOURCE_LOOP5,
)
from workflows.academic_lit_review.state import (
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
    loop_progress = state["loop_progress"]
    max_iterations = loop_progress["max_iterations_per_loop"]
    topic = state["input"].get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_supervision(
        final_review=state["current_review"],
        paper_corpus=state["paper_corpus"],
        paper_summaries=state["paper_summaries"],
        clusters=state["clusters"],
        quality_settings=state["quality_settings"],
        input_data=state["input"],
        zotero_keys=state["zotero_keys"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop1_theory:{topic}",
        },
    )

    after_text = result.get("final_review_v2", state["current_review"])
    iterations_used = result.get("iterations", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_1"] = iterations_used
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

    # Track citation key sources for any new papers added
    zotero_key_sources = state.get("zotero_key_sources", {})
    added_keys = result.get("added_zotero_keys", {})
    for doi, key in added_keys.items():
        if key not in zotero_key_sources:
            zotero_key_sources[key] = {
                "key": key,
                "source": CITATION_SOURCE_LOOP1,
                "verified": True,  # Added through workflow, assumed valid
                "doi": doi,
            }

    logger.info(f"Loop 1 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "review_loop1": after_text,
        "loop1_result": result,
        "loop_progress": loop_progress,
        "paper_corpus": result.get("added_papers", {}),
        "paper_summaries": result.get("added_summaries", {}),
        "zotero_key_sources": zotero_key_sources,
    }


async def run_loop2_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 2: Literature base expansion."""
    logger.info("Running Loop 2: Literature base expansion")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    max_iterations = loop_progress["max_iterations_per_loop"]
    topic = state["input"].get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop2_standalone(
        review=state["current_review"],
        paper_corpus=state["paper_corpus"],
        paper_summaries=state["paper_summaries"],
        zotero_keys=state["zotero_keys"],
        input_data=state["input"],
        quality_settings=state["quality_settings"],
        max_iterations=max_iterations,
        config={
            "run_id": loop_run_id,
            "run_name": f"loop2_literature:{topic}",
        },
    )

    after_text = result.get("current_review", state["current_review"])
    iterations_used = result.get("iteration", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_2"] = iterations_used
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

    # Track citation key sources for any new papers added
    zotero_key_sources = state.get("zotero_key_sources", {})
    new_zotero_keys = result.get("zotero_keys", {})
    for doi, key in new_zotero_keys.items():
        if key not in zotero_key_sources:
            zotero_key_sources[key] = {
                "key": key,
                "source": CITATION_SOURCE_LOOP2,
                "verified": True,
                "doi": doi,
            }

    logger.info(f"Loop 2 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "review_loop2": after_text,
        "loop2_result": result,
        "loop_progress": loop_progress,
        "paper_corpus": result.get("paper_corpus", state["paper_corpus"]),
        "paper_summaries": result.get("paper_summaries", state["paper_summaries"]),
        "zotero_keys": result.get("zotero_keys", state["zotero_keys"]),
        "zotero_key_sources": zotero_key_sources,
    }


async def run_loop3_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 3: Structure and cohesion."""
    logger.info("Running Loop 3: Structure and cohesion")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    max_iterations = loop_progress["max_iterations_per_loop"]
    topic = state["input"].get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop3_standalone(
        review=state["current_review"],
        input_data=state["input"],
        max_iterations=max_iterations,
        config={
            "run_id": loop_run_id,
            "run_name": f"loop3_structure:{topic}",
        },
    )

    after_text = result.get("current_review", state["current_review"])
    iterations_used = result.get("iteration", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_3"] = iterations_used
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
        "review_loop3": after_text,
        "loop3_result": result,
        "loop_progress": loop_progress,
        "paper_summaries": state["paper_summaries"],
        "zotero_keys": state["zotero_keys"],
    }


async def run_loop4_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 4: Section-level deep editing with optional Zotero verification."""
    logger.info("Running Loop 4: Section-level deep editing")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    max_iterations = loop_progress["max_iterations_per_loop"]
    topic = state["input"].get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    # Check if Zotero verification is enabled
    verify_zotero = state.get("verify_zotero", False)

    result = await run_loop4_standalone(
        review=state["current_review"],
        paper_summaries=state["paper_summaries"],
        input_data=state["input"],
        zotero_keys=state["zotero_keys"],
        max_iterations=max_iterations,
        config={
            "run_id": loop_run_id,
            "run_name": f"loop4_editing:{topic}",
        },
        verify_zotero=verify_zotero,
    )

    after_text = result.get("edited_review", state["current_review"])
    iterations_used = result.get("iterations", 0)

    # Update loop progress
    loop_progress["loop_iterations"]["loop_4"] = iterations_used
    loop_progress["current_loop"] = 4.5

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

    # Track verified citation keys
    zotero_key_sources = state.get("zotero_key_sources", {})
    verified_keys = result.get("verified_citation_keys", set())
    for key in verified_keys:
        if key not in zotero_key_sources:
            zotero_key_sources[key] = {
                "key": key,
                "source": CITATION_SOURCE_LOOP4,
                "verified": True,
                "doi": None,
            }

    logger.info(f"Loop 4 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "review_loop4": after_text,
        "loop4_result": result,
        "loop_progress": loop_progress,
        "paper_summaries": state["paper_summaries"],
        "zotero_keys": state["zotero_keys"],
        "zotero_key_sources": zotero_key_sources,
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
        "paper_summaries": state["paper_summaries"],
        "zotero_keys": state["zotero_keys"],
    }


async def run_loop5_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 5: Fact and reference checking with optional Zotero verification."""
    logger.info("Running Loop 5: Fact and reference checking")

    before_text = state["current_review"]
    topic = state["input"].get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    # Check if Zotero verification is enabled
    verify_zotero = state.get("verify_zotero", False)

    result = await run_loop5_standalone(
        review=state["current_review"],
        paper_summaries=state["paper_summaries"],
        zotero_keys=state["zotero_keys"],
        max_iterations=1,
        config={
            "run_id": loop_run_id,
            "run_name": f"loop5_factcheck:{topic}",
        },
        topic=topic,
        verify_zotero=verify_zotero,
    )

    after_text = result.get("current_review", state["current_review"])

    loop_progress = state["loop_progress"]
    loop_progress["loop_iterations"]["loop_5"] = 1
    loop_progress["current_loop"] = 6

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

    # Track verified citation keys
    zotero_key_sources = state.get("zotero_key_sources", {})
    verified_keys = result.get("verified_citation_keys", set())
    for key in verified_keys:
        if key not in zotero_key_sources:
            zotero_key_sources[key] = {
                "key": key,
                "source": CITATION_SOURCE_LOOP5,
                "verified": True,
                "doi": None,
            }

    logger.info(
        f"Loop 5 complete: {len(human_review_items)} items flagged for human review"
    )

    return {
        "current_review": after_text,
        "loop5_result": result,
        "loop_progress": loop_progress,
        "human_review_items": human_review_items,
        "zotero_key_sources": zotero_key_sources,
    }


def finalize_node(state: OrchestrationState) -> dict[str, Any]:
    """Finalize orchestration and prepare final outputs."""
    logger.info("Finalizing multi-loop orchestration")

    loop_progress = state["loop_progress"]
    total_iterations = sum(loop_progress["loop_iterations"].values())

    completion_reason = (
        f"All loops complete. Total iterations: {total_iterations}. "
        f"Max per loop: {loop_progress['max_iterations_per_loop']}"
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
    """Route after Loop 4.5 cohesion check."""
    loop4_5_result = state.get("loop4_5_result", {})
    needs_restructuring = loop4_5_result.get("needs_restructuring", False)
    repeat_count = state.get("loop3_repeat_count", 0)

    max_repeats = state["loop_progress"]["max_iterations_per_loop"]
    if needs_restructuring and repeat_count < max_repeats:
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
    """Create the multi-loop orchestration graph."""
    builder = StateGraph(OrchestrationState)

    builder.add_node("loop1", run_loop1_node)
    builder.add_node("loop2", run_loop2_node)
    builder.add_node("loop3", run_loop3_node)
    builder.add_node("loop4", run_loop4_node)
    builder.add_node("loop4_5", run_loop4_5_node)
    builder.add_node("increment_and_loop3", increment_loop3_repeat_node)
    builder.add_node("loop5", run_loop5_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "loop1")
    builder.add_edge("loop1", "loop2")
    builder.add_edge("loop2", "loop3")
    builder.add_edge("loop3", "loop4")
    builder.add_edge("loop4", "loop4_5")

    builder.add_conditional_edges(
        "loop4_5",
        route_after_loop4_5,
        {
            "increment_and_loop3": "increment_and_loop3",
            "loop5": "loop5",
        },
    )

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
    max_iterations_per_loop: int = 3,
    verify_zotero: bool = False,
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
        max_iterations_per_loop: Max iterations each loop can use
        verify_zotero: If True, verify new citations against Zotero programmatically

    Returns:
        Dictionary containing final_review, loop_progress, human_review_items,
        completion_reason, and individual loop results
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
        max_iterations_per_loop=max_iterations_per_loop,
        checkpoints=[],
        revision_history=[],
        loop3_repeat_count=0,
    )

    # Initialize citation key source tracking
    zotero_key_sources = {}
    for doi, key in zotero_keys.items():
        zotero_key_sources[key] = {
            "key": key,
            "source": CITATION_SOURCE_INITIAL,
            "verified": True,
            "doi": doi,
        }

    initial_state = OrchestrationState(
        current_review=review,
        final_review=None,
        input=input_data,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        clusters=clusters,
        quality_settings=quality_settings,
        zotero_keys=zotero_keys,
        zotero_key_sources=zotero_key_sources,
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
        verify_zotero=verify_zotero,
    )

    graph = create_orchestration_graph()
    topic = input_data.get("topic", "")[:20]
    orch_run_id = uuid.uuid4()
    logger.info(f"Starting supervision orchestration, LangSmith run ID: {orch_run_id}")
    final_state = await graph.ainvoke(
        initial_state,
        config={
            "run_id": orch_run_id,
            "run_name": f"supervision:{topic}",
        },
    )

    logger.info(
        f"Orchestration complete: {final_state.get('completion_reason', 'Unknown')}"
    )

    return {
        "final_review": final_state.get("final_review", review),
        "review_loop1": final_state.get("review_loop1"),
        "review_loop2": final_state.get("review_loop2"),
        "review_loop3": final_state.get("review_loop3"),
        "review_loop4": final_state.get("review_loop4"),
        "loop_progress": final_state.get("loop_progress", loop_progress),
        "human_review_items": final_state.get("human_review_items", []),
        "completion_reason": final_state.get("completion_reason", ""),
        "zotero_key_sources": final_state.get("zotero_key_sources", {}),
        "loop1_result": final_state.get("loop1_result"),
        "loop2_result": final_state.get("loop2_result"),
        "loop3_result": final_state.get("loop3_result"),
        "loop4_result": final_state.get("loop4_result"),
        "loop4_5_result": final_state.get("loop4_5_result"),
        "loop5_result": final_state.get("loop5_result"),
    }


# Valid supervision_loops config values
SUPERVISION_LOOP_CONFIGS = {
    "none": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "all": 5,
    "five": 5,
}


async def run_supervision_configurable(
    review: str,
    paper_corpus: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    clusters: list,
    input_data: dict,
    quality_settings: dict,
    max_iterations_per_loop: int = 3,
    loops: str = "all",
    verify_zotero: bool = False,
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
        max_iterations_per_loop: Max iterations each loop can use
        loops: Which loops to run - "none", "one", "two", "three", "four", "all"
        verify_zotero: If True, verify new citations against Zotero programmatically

    Returns:
        Dictionary containing final_review, loop_progress, human_review_items,
        completion_reason, and loops_run
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
            max_iterations_per_loop=max_iterations_per_loop,
            verify_zotero=verify_zotero,
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
    topic = input_data.get("topic", "")[:20]

    review_loop1 = None
    review_loop2 = None
    review_loop3 = None
    review_loop4 = None

    # Loop 1: Theoretical Depth
    if loop_count >= 1:
        logger.info("Running Loop 1: Theoretical depth expansion")
        loops_run.append("loop1")
        loop1_run_id = uuid.uuid4()

        loop1_result = await run_supervision(
            final_review=current_review,
            paper_corpus=current_corpus,
            paper_summaries=current_summaries,
            clusters=clusters,
            quality_settings=quality_settings,
            input_data=input_data,
            zotero_keys=current_zotero,
            config={
                "run_id": loop1_run_id,
                "run_name": f"loop1_theory:{topic}",
            },
        )

        current_review = loop1_result.get("final_review_v2", current_review)
        review_loop1 = current_review
        all_results["loop1_result"] = loop1_result

        added_papers = loop1_result.get("added_papers", {})
        added_summaries = loop1_result.get("added_summaries", {})
        current_corpus.update(added_papers)
        current_summaries.update(added_summaries)

    # Loop 2: Literature Base Expansion
    if loop_count >= 2:
        logger.info("Running Loop 2: Literature base expansion")
        loops_run.append("loop2")
        loop2_run_id = uuid.uuid4()

        loop2_result = await run_loop2_standalone(
            review=current_review,
            paper_corpus=current_corpus,
            paper_summaries=current_summaries,
            zotero_keys=current_zotero,
            input_data=input_data,
            quality_settings=quality_settings,
            max_iterations=max_iterations_per_loop,
            config={
                "run_id": loop2_run_id,
                "run_name": f"loop2_literature:{topic}",
            },
        )

        current_review = loop2_result.get("current_review", current_review)
        review_loop2 = current_review
        current_corpus = loop2_result.get("paper_corpus", current_corpus)
        current_summaries = loop2_result.get("paper_summaries", current_summaries)
        current_zotero = loop2_result.get("zotero_keys", current_zotero)
        all_results["loop2_result"] = loop2_result

    # Loop 3: Structure and Cohesion
    if loop_count >= 3:
        logger.info("Running Loop 3: Structure and cohesion")
        loops_run.append("loop3")
        loop3_run_id = uuid.uuid4()

        loop3_result = await run_loop3_standalone(
            review=current_review,
            input_data=input_data,
            max_iterations=max_iterations_per_loop,
            config={
                "run_id": loop3_run_id,
                "run_name": f"loop3_structure:{topic}",
            },
        )

        current_review = loop3_result.get("current_review", current_review)
        review_loop3 = current_review
        all_results["loop3_result"] = loop3_result

    # Loop 4 + 4.5: Section Editing + Cohesion Check
    if loop_count >= 4:
        logger.info("Running Loop 4: Section-level deep editing")
        loops_run.append("loop4")
        loop4_run_id = uuid.uuid4()

        loop4_result = await run_loop4_standalone(
            review=current_review,
            paper_summaries=current_summaries,
            input_data=input_data,
            zotero_keys=current_zotero,
            max_iterations=max_iterations_per_loop,
            config={
                "run_id": loop4_run_id,
                "run_name": f"loop4_editing:{topic}",
            },
            verify_zotero=verify_zotero,
        )

        current_review = loop4_result.get("edited_review", current_review)
        review_loop4 = current_review
        all_results["loop4_result"] = loop4_result

        # Run Loop 4.5 cohesion check
        logger.info("Running Loop 4.5: Cohesion check")
        loops_run.append("loop4_5")

        cohesion_result = await check_cohesion(current_review)
        all_results["loop4_5_result"] = {
            "needs_restructuring": cohesion_result.needs_restructuring,
            "reasoning": cohesion_result.reasoning,
        }

        if cohesion_result.needs_restructuring:
            logger.info("Cohesion check flagged issues - running Loop 3 again")
            loop3_repeat_run_id = uuid.uuid4()
            loop3_repeat = await run_loop3_standalone(
                review=current_review,
                input_data=input_data,
                max_iterations=2,
                config={
                    "run_id": loop3_repeat_run_id,
                    "run_name": f"loop3_repeat:{topic}",
                },
            )
            current_review = loop3_repeat.get("current_review", current_review)
            review_loop3 = current_review

    completion_reason = f"Completed {loop_count} supervision loop(s)"
    logger.info(f"Partial supervision complete: {completion_reason}")

    return {
        "final_review": current_review,
        "review_loop1": review_loop1,
        "review_loop2": review_loop2,
        "review_loop3": review_loop3,
        "review_loop4": review_loop4,
        "loop_progress": None,
        "human_review_items": human_review_items,
        "completion_reason": completion_reason,
        "loops_run": loops_run,
        "paper_corpus": current_corpus,
        "paper_summaries": current_summaries,
        "zotero_keys": current_zotero,
        **all_results,
    }
