"""API functions for supervision orchestration."""

import logging
import uuid

from .types import OrchestrationState
from .builder import create_orchestration_graph
from ..graph import run_loop1_standalone
from ..loops.loop2 import run_loop2_standalone
from ..loops.loop3 import run_loop3_standalone
from ..loops.loop4_editing import run_loop4_standalone
from ..loops.loop4_5_cohesion import check_cohesion
from ..utils.citation_validation import CITATION_SOURCE_INITIAL
from workflows.research.academic_lit_review.state import MultiLoopProgress

logger = logging.getLogger(__name__)

SUPERVISION_LOOP_CONFIGS = {
    "none": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "all": 5,
    "five": 5,
}


async def run_supervision_orchestration(
    review: str,
    paper_corpus: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    clusters: list,
    input_data: dict,
    quality_settings: dict,
    max_iterations_per_loop: int = 3,
    verify_zotero: bool = True,
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
        revision_history=[],
        loop3_repeat_count=0,
    )

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
        loop_errors=[],
    )

    graph = create_orchestration_graph()
    topic = input_data.get("topic", "")[:20]
    orch_run_id = uuid.uuid4()
    logger.info(f"Starting supervision orchestration for '{topic}' (run_id: {orch_run_id})")
    final_state = await graph.ainvoke(
        initial_state,
        config={
            "run_id": orch_run_id,
            "run_name": f"supervision:{topic}",
        },
    )

    logger.info(f"Orchestration complete: {final_state.get('completion_reason', 'Unknown')}")

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
        "loop_errors": final_state.get("loop_errors", []),
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
    verify_zotero: bool = True,
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
        logger.debug("Supervision skipped (loops='none')")
        return {
            "final_review": review,
            "loop_progress": None,
            "human_review_items": [],
            "completion_reason": "Supervision disabled",
            "loops_run": [],
        }

    if loop_count >= 5:
        logger.info("Running full multi-loop supervision")
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

    logger.info(f"Running {loop_count} supervision loop(s)")

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

    if loop_count >= 1:
        logger.info("Running Loop 1: Theoretical depth")
        loops_run.append("loop1")
        loop1_run_id = uuid.uuid4()

        loop1_result = await run_loop1_standalone(
            review=current_review,
            topic=input_data.get("topic", ""),
            research_questions=input_data.get("research_questions", []),
            max_iterations=max_iterations_per_loop,
            source_count=len(current_corpus),
            quality_settings=quality_settings,
            config={
                "run_id": loop1_run_id,
                "run_name": f"loop1_theory:{topic}",
            },
        )

        current_review = loop1_result.current_review
        review_loop1 = current_review
        all_results["loop1_result"] = {
            "issues_explored": loop1_result.issues_explored,
            "changes_summary": loop1_result.changes_summary,
        }

    if loop_count >= 2:
        logger.info("Running Loop 2: Literature expansion")
        loops_run.append("loop2")
        loop2_run_id = uuid.uuid4()

        loop2_result = await run_loop2_standalone(
            review=current_review,
            topic=input_data.get("topic", ""),
            research_questions=input_data.get("research_questions", []),
            quality_settings=quality_settings,
            config={
                "run_id": loop2_run_id,
                "run_name": f"loop2_literature:{topic}",
            },
        )

        current_review = loop2_result.current_review
        review_loop2 = current_review
        all_results["loop2_result"] = {
            "explored_bases": loop2_result.explored_bases,
            "changes_summary": loop2_result.changes_summary,
        }

    if loop_count >= 3:
        logger.info("Running Loop 3: Structure and cohesion")
        loops_run.append("loop3")
        loop3_run_id = uuid.uuid4()

        loop3_result = await run_loop3_standalone(
            review=current_review,
            topic=input_data.get("topic", ""),
            quality_settings=quality_settings,
            config={
                "run_id": loop3_run_id,
                "run_name": f"loop3_structure:{topic}",
            },
        )

        current_review = loop3_result.current_review
        review_loop3 = current_review
        all_results["loop3_result"] = {
            "iterations_used": loop3_result.iterations_used,
            "changes_summary": loop3_result.changes_summary,
        }

    if loop_count >= 4:
        logger.info("Running Loop 4: Section editing")
        loops_run.append("loop4")
        loop4_run_id = uuid.uuid4()

        loop4_result = await run_loop4_standalone(
            review=current_review,
            topic=input_data.get("topic", ""),
            quality_settings=quality_settings,
            config={
                "run_id": loop4_run_id,
                "run_name": f"loop4_editing:{topic}",
            },
        )

        current_review = loop4_result.current_review
        review_loop4 = current_review
        all_results["loop4_result"] = {
            "iterations_used": loop4_result.iterations_used,
            "changes_summary": loop4_result.changes_summary,
        }

        logger.info("Running Loop 4.5: Cohesion check")
        loops_run.append("loop4_5")

        cohesion_result = await check_cohesion(current_review)
        all_results["loop4_5_result"] = {
            "needs_restructuring": cohesion_result.needs_restructuring,
            "reasoning": cohesion_result.reasoning,
        }

        if cohesion_result.needs_restructuring:
            logger.info("Cohesion check failed, repeating Loop 3")
            loop3_repeat_run_id = uuid.uuid4()
            # Use reduced quality settings for repeat (max 2 iterations)
            repeat_quality = {**quality_settings, "max_stages": 1}
            loop3_repeat = await run_loop3_standalone(
                review=current_review,
                topic=input_data.get("topic", ""),
                quality_settings=repeat_quality,
                config={
                    "run_id": loop3_repeat_run_id,
                    "run_name": f"loop3_repeat:{topic}",
                },
            )
            current_review = loop3_repeat.current_review
            review_loop3 = current_review

    completion_reason = f"Completed {loop_count} supervision loop(s)"
    logger.info(completion_reason)

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
