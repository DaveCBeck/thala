"""Loop node functions for supervision orchestration."""

import logging
import uuid
from typing import Any

from .types import OrchestrationState
from ..graph import run_supervision
from ..loops.loop2 import run_loop2_standalone
from ..loops.loop3 import run_loop3_standalone
from ..loops.loop4_editing import run_loop4_standalone
from ..loops.loop4_5_cohesion import check_cohesion
from ..loops.loop5_factcheck import run_loop5_standalone
from ..utils import document_revision
from ..utils.citation_validation import (
    CITATION_SOURCE_LOOP1,
    CITATION_SOURCE_LOOP2,
    CITATION_SOURCE_LOOP4,
    CITATION_SOURCE_LOOP5,
)
from workflows.academic_lit_review.state import LoopCheckpoint

logger = logging.getLogger(__name__)


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

    checkpoint = LoopCheckpoint(
        loop_number=1,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

    zotero_key_sources = state.get("zotero_key_sources", {})
    updated_zotero_keys = dict(state["zotero_keys"])
    added_keys = result.get("added_zotero_keys", {})
    for doi, key in added_keys.items():
        if key not in zotero_key_sources:
            zotero_key_sources[key] = {
                "key": key,
                "source": CITATION_SOURCE_LOOP1,
                "verified": True,
                "doi": doi,
            }
        if doi not in updated_zotero_keys:
            updated_zotero_keys[doi] = key

    logger.info(f"Loop 1 complete: {iterations_used} iterations used, {len(added_keys)} new citation keys")

    loop_errors = state.get("loop_errors", [])
    if result.get("loop_error"):
        loop_errors.append(result["loop_error"])
    if result.get("errors"):
        loop_errors.extend(result["errors"])

    iterations_failed = result.get("iterations_failed", 0)
    if iterations_failed > 0:
        logger.warning(f"Loop 1: {iterations_failed} iterations failed")

    return {
        "current_review": after_text,
        "review_loop1": after_text,
        "loop1_result": result,
        "loop_progress": loop_progress,
        "paper_corpus": result.get("added_papers", {}),
        "paper_summaries": result.get("added_summaries", {}),
        "zotero_keys": updated_zotero_keys,
        "zotero_key_sources": zotero_key_sources,
        "loop_errors": loop_errors,
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

    checkpoint = LoopCheckpoint(
        loop_number=2,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

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

    loop_errors = state.get("loop_errors", [])
    if result.get("errors"):
        loop_errors.extend(result["errors"])

    iterations_failed = result.get("iterations_failed", 0)
    if iterations_failed > 0:
        logger.warning(f"Loop 2: {iterations_failed} iterations failed")

    return {
        "current_review": after_text,
        "review_loop2": after_text,
        "loop2_result": result,
        "loop_progress": loop_progress,
        "paper_corpus": result.get("paper_corpus", state["paper_corpus"]),
        "paper_summaries": result.get("paper_summaries", state["paper_summaries"]),
        "zotero_keys": result.get("zotero_keys", state["zotero_keys"]),
        "zotero_key_sources": zotero_key_sources,
        "loop_errors": loop_errors,
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

    verify_zotero = state.get("verify_zotero", True)

    zotero_key_sources = state.get("zotero_key_sources", {})

    result = await run_loop4_standalone(
        review=state["current_review"],
        paper_summaries=state["paper_summaries"],
        input_data=state["input"],
        zotero_keys=state["zotero_keys"],
        zotero_key_sources=zotero_key_sources,
        max_iterations=max_iterations,
        config={
            "run_id": loop_run_id,
            "run_name": f"loop4_editing:{topic}",
        },
        verify_zotero=verify_zotero,
    )

    after_text = result.get("edited_review", state["current_review"])
    iterations_used = result.get("iterations", 0)

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

    checkpoint = LoopCheckpoint(
        loop_number=4,
        iteration=iterations_used,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

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

    verify_zotero = state.get("verify_zotero", True)

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

    if before_text != after_text:
        revision = await document_revision(
            loop_number=5,
            iteration=1,
            before_text=before_text,
            after_text=after_text,
        )
        loop_progress["revision_history"].append(revision)

    checkpoint = LoopCheckpoint(
        loop_number=5,
        iteration=1,
        review_snapshot=after_text[:1000] + "...",
        timestamp="",
    )
    loop_progress["checkpoints"].append(checkpoint)

    human_review_items = result.get("human_review_items", [])

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
    }
