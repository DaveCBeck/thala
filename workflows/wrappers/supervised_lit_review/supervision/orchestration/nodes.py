"""Loop node functions for supervision orchestration."""

import logging
import uuid
from typing import Any

from .types import OrchestrationState
from ..graph import run_loop1_standalone
from ..loops.loop2 import run_loop2_standalone
from ..loops.loop3 import run_loop3_standalone
from ..loops.loop4_editing import run_loop4_standalone
from ..loops.loop4_5_cohesion import check_cohesion
from ..loops.loop5_factcheck import run_loop5_standalone
from ..utils import document_revision

logger = logging.getLogger(__name__)


async def run_loop1_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 1: Theoretical depth expansion."""
    logger.info("Running Loop 1: Theoretical depth expansion")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    max_iterations = loop_progress["max_iterations_per_loop"]
    input_data = state["input"]
    topic = input_data.get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop1_standalone(
        review=state["current_review"],
        topic=input_data.get("topic", ""),
        research_questions=input_data.get("research_questions", []),
        max_iterations=max_iterations,
        source_count=len(state["paper_corpus"]),
        quality_settings=state["quality_settings"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop1_theory:{topic}",
        },
    )

    after_text = result.current_review
    iterations_used = len(result.issues_explored)

    loop_progress["loop_iterations"]["loop_1"] = iterations_used
    loop_progress["current_loop"] = 2

    if before_text != after_text:
        revision = await document_revision(
            loop_number=1,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    logger.info(f"Loop 1 complete: {iterations_used} issues explored")

    return {
        "current_review": after_text,
        "review_loop1": after_text,
        "loop1_result": {
            "issues_explored": result.issues_explored,
            "changes_summary": result.changes_summary,
        },
        "loop_progress": loop_progress,
    }


async def run_loop2_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 2: Literature base expansion."""
    logger.info("Running Loop 2: Literature base expansion")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    input_data = state["input"]
    topic = input_data.get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop2_standalone(
        review=state["current_review"],
        topic=input_data.get("topic", ""),
        research_questions=input_data.get("research_questions", []),
        quality_settings=state["quality_settings"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop2_literature:{topic}",
        },
    )

    after_text = result.current_review
    iterations_used = len(result.explored_bases)

    loop_progress["loop_iterations"]["loop_2"] = iterations_used
    loop_progress["current_loop"] = 3

    if before_text != after_text:
        revision = await document_revision(
            loop_number=2,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    logger.info(f"Loop 2 complete: {iterations_used} bases explored")

    return {
        "current_review": after_text,
        "review_loop2": after_text,
        "loop2_result": {
            "explored_bases": result.explored_bases,
            "changes_summary": result.changes_summary,
        },
        "loop_progress": loop_progress,
    }


async def run_loop3_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 3: Structure and cohesion."""
    logger.info("Running Loop 3: Structure and cohesion")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    input_data = state["input"]
    topic = input_data.get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop3_standalone(
        review=state["current_review"],
        topic=input_data.get("topic", ""),
        quality_settings=state["quality_settings"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop3_structure:{topic}",
        },
    )

    after_text = result.current_review
    iterations_used = result.iterations_used

    loop_progress["loop_iterations"]["loop_3"] = iterations_used
    loop_progress["current_loop"] = 4

    if before_text != after_text:
        revision = await document_revision(
            loop_number=3,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    logger.info(f"Loop 3 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "review_loop3": after_text,
        "loop3_result": {
            "iterations_used": result.iterations_used,
            "changes_summary": result.changes_summary,
        },
        "loop_progress": loop_progress,
        "paper_summaries": state["paper_summaries"],
        "zotero_keys": state["zotero_keys"],
        "zotero_key_sources": state.get("zotero_key_sources", {}),
    }


async def run_loop4_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 4: Section-level deep editing."""
    logger.info("Running Loop 4: Section-level deep editing")

    before_text = state["current_review"]
    loop_progress = state["loop_progress"]
    input_data = state["input"]
    topic = input_data.get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop4_standalone(
        review=state["current_review"],
        topic=input_data.get("topic", ""),
        quality_settings=state["quality_settings"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop4_editing:{topic}",
        },
    )

    after_text = result.current_review
    iterations_used = result.iterations_used

    loop_progress["loop_iterations"]["loop_4"] = iterations_used
    loop_progress["current_loop"] = 4.5

    if before_text != after_text:
        revision = await document_revision(
            loop_number=4,
            iteration=iterations_used,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    logger.info(f"Loop 4 complete: {iterations_used} iterations used")

    return {
        "current_review": after_text,
        "review_loop4": after_text,
        "loop4_result": {
            "iterations_used": result.iterations_used,
            "changes_summary": result.changes_summary,
        },
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
        "paper_summaries": state["paper_summaries"],
        "zotero_keys": state["zotero_keys"],
        "zotero_key_sources": state.get("zotero_key_sources", {}),
    }


async def run_loop5_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 5: Fact and reference checking."""
    logger.info("Running Loop 5: Fact and reference checking")

    before_text = state["current_review"]
    input_data = state["input"]
    topic = input_data.get("topic", "")[:20]
    loop_run_id = uuid.uuid4()

    result = await run_loop5_standalone(
        review=state["current_review"],
        topic=input_data.get("topic", ""),
        quality_settings=state["quality_settings"],
        config={
            "run_id": loop_run_id,
            "run_name": f"loop5_factcheck:{topic}",
        },
    )

    after_text = result.current_review

    loop_progress = state["loop_progress"]
    loop_progress["loop_iterations"]["loop_5"] = 1
    loop_progress["current_loop"] = 6

    if before_text != after_text:
        revision = await document_revision(
            loop_number=5,
            iteration=1,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    logger.info(f"Loop 5 complete: {result.changes_summary}")

    return {
        "current_review": after_text,
        "loop5_result": {
            "changes_summary": result.changes_summary,
            "human_review_items": result.human_review_items,
        },
        "loop_progress": loop_progress,
        "human_review_items": result.human_review_items,
    }


def finalize_node(state: OrchestrationState) -> dict[str, Any]:
    """Finalize orchestration and prepare final outputs."""
    logger.info("Finalizing multi-loop orchestration")

    loop_progress = state["loop_progress"]
    total_iterations = sum(loop_progress["loop_iterations"].values())

    loop_errors = state.get("loop_errors", [])
    error_count = len(loop_errors)

    if error_count > 0:
        error_summary = f" ({error_count} errors encountered)"
        loop1_errors = [e for e in loop_errors if e.get("loop_number") == 1]
        loop2_errors = [e for e in loop_errors if e.get("loop_number") == 2]
        if loop1_errors:
            logger.warning(f"Loop 1 had {len(loop1_errors)} errors")
        if loop2_errors:
            logger.warning(f"Loop 2 had {len(loop2_errors)} errors")
    else:
        error_summary = ""

    completion_reason = (
        f"All loops complete. Total iterations: {total_iterations}. "
        f"Max per loop: {loop_progress['max_iterations_per_loop']}{error_summary}"
    )

    return {
        "final_review": state["current_review"],
        "is_complete": True,
        "completion_reason": completion_reason,
        "loop_errors": loop_errors,
        "zotero_keys": state.get("zotero_keys", {}),
        "zotero_key_sources": state.get("zotero_key_sources", {}),
    }
