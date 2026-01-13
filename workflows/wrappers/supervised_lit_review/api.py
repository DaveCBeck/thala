"""Main entry point for supervised academic literature review workflow.

This wrapper runs the academic_lit_review workflow first, then applies
configurable supervision loops to enhance quality.
"""

import logging
from datetime import datetime
from typing import Any, Literal, Optional

from workflows.research.academic_lit_review.graph.api import academic_lit_review
from workflows.research.academic_lit_review.quality_presets import QUALITY_PRESETS
from workflows.shared.workflow_state_store import load_workflow_state, save_workflow_state
from workflows.wrappers.supervised_lit_review.supervision.orchestration import (
    run_supervision_configurable,
)

logger = logging.getLogger(__name__)


async def supervised_lit_review(
    topic: str,
    research_questions: list[str],
    quality: Literal["test", "quick", "standard", "comprehensive", "high_quality"] = "standard",
    language: str = "en",
    date_range: Optional[tuple[int, int]] = None,
    supervision_loops: str = "all",
) -> dict[str, Any]:
    """Run a complete academic literature review WITH supervision loops.

    This function first runs the core literature review workflow, then applies
    multi-loop supervision to enhance the quality of the review.

    Args:
        topic: Research topic (e.g., "Large Language Models in Scientific Discovery")
        research_questions: List of specific questions to address
        quality: Quality tier - "quick", "standard", "comprehensive", "high_quality"
        language: ISO 639-1 language code (default: "en")
        date_range: Optional (start_year, end_year) filter
        supervision_loops: Which loops to run - "none", "one", "two", "three", "four", "all"

    Returns:
        Dict containing:
        - final_review: Complete literature review (after supervision if enabled)
        - final_review_v2: Supervised version (if supervision ran)
        - paper_corpus: All discovered papers (including any added by supervision)
        - paper_summaries: Processed paper summaries
        - clusters: Thematic clusters
        - references: Formatted citations
        - supervision: Supervision metadata (loops run, progress)
        - human_review_items: Items flagged for human review
        - errors: Any errors encountered

    Example:
        result = await supervised_lit_review(
            topic="Large Language Models in Scientific Discovery",
            research_questions=[
                "How are LLMs being used for hypothesis generation?",
                "What are the methodological challenges of using LLMs in research?",
            ],
            quality="high_quality",
            supervision_loops="all",
        )

        # Access the supervised review
        print(f"Original review: {len(result['final_review'].split())} words")
        if result.get('final_review_v2'):
            print(f"Supervised review: {len(result['final_review_v2'].split())} words")
    """
    logger.info(f"Starting supervised literature review for '{topic}' with {supervision_loops} loops")

    # Step 1: Run the core literature review (without supervision)
    lit_review_result = await academic_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        language=language,
        date_range=date_range,
    )

    # Check for errors in lit review
    has_errors = bool(lit_review_result.get("errors"))
    if has_errors:
        logger.warning(f"Base literature review completed with {len(lit_review_result['errors'])} errors")

    final_review = lit_review_result.get("final_review", "")

    # A valid review should be substantial (at least 500 chars)
    # If we have errors and a tiny "review", it's not a real review
    has_valid_review = final_review and len(final_review) >= 500

    # If no supervision requested, no valid review, or lit review failed - return as-is
    if supervision_loops == "none" or not has_valid_review:
        if has_errors and not has_valid_review:
            logger.error("Base literature review failed - skipping supervision")
        else:
            logger.debug("Supervision disabled or no valid review content")
        # Determine status: failed if no valid review, otherwise use lit_review status
        if has_errors and not has_valid_review:
            status = "failed"
        else:
            status = lit_review_result.get("status", "success" if has_valid_review else "failed")
        return {
            "final_report": lit_review_result.get("final_report", final_review),
            "status": status,
            "langsmith_run_id": lit_review_result.get("langsmith_run_id"),
            "errors": lit_review_result.get("errors", []),
            "source_count": lit_review_result.get("source_count", 0),
            "started_at": lit_review_result.get("started_at"),
            "completed_at": lit_review_result.get("completed_at"),
        }

    # Step 2: Apply supervision loops
    logger.info(f"Starting supervision on {len(final_review)} char review")

    # Get quality settings for supervision
    quality_preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
    max_stages = quality_preset.get("max_stages", 3)

    # Build input data for supervision
    from workflows.research.academic_lit_review.state import LitReviewInput

    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        language_code=language,
    )

    try:
        # Load full state from state store (required for supervision)
        run_id = lit_review_result.get("langsmith_run_id")
        full_state = load_workflow_state("academic_lit_review", run_id) if run_id else None

        if not full_state:
            logger.error(f"Cannot load state for run {run_id} - supervision requires state store (THALA_MODE=dev)")
            return {
                "final_report": lit_review_result.get("final_report", final_review),
                "status": "partial",
                "langsmith_run_id": run_id,
                "errors": lit_review_result.get("errors", []) + [
                    {"phase": "supervision", "error": "State store not available - run with THALA_MODE=dev"}
                ],
                "source_count": lit_review_result.get("source_count", 0),
                "started_at": lit_review_result.get("started_at"),
                "completed_at": datetime.utcnow(),
            }

        logger.debug(f"Loaded workflow state from run {run_id}")
        paper_corpus = full_state.get("paper_corpus", {})
        paper_summaries = full_state.get("paper_summaries", {})
        zotero_keys = full_state.get("zotero_keys", {})
        clusters = full_state.get("clusters", [])

        supervision_result = await run_supervision_configurable(
            review=final_review,
            paper_corpus=paper_corpus,
            paper_summaries=paper_summaries,
            zotero_keys=zotero_keys,
            clusters=clusters,
            input_data=input_data,
            quality_settings=quality_preset,
            max_iterations_per_loop=max_stages,
            loops=supervision_loops,
        )

        final_review_v2 = supervision_result.get("final_review", final_review)
        loops_run = supervision_result.get("loops_run", [])
        human_review_items = supervision_result.get("human_review_items", [])
        completion_reason = supervision_result.get("completion_reason", "")

        # Merge new papers from supervision
        added_papers = supervision_result.get("paper_corpus", {})
        added_summaries = supervision_result.get("paper_summaries", {})
        # Use the corpus we loaded (from state store or return dict)
        original_corpus = paper_corpus
        original_summaries = paper_summaries

        merged_corpus = {**original_corpus, **added_papers}
        merged_summaries = {**original_summaries, **added_summaries}

        new_paper_count = len(added_papers) - len(
            set(added_papers.keys()) & set(original_corpus.keys())
        )

        logger.info(
            f"Supervision complete: {len(loops_run)} loops, "
            f"{new_paper_count} papers added, {len(human_review_items)} items flagged. "
            f"Reason: {completion_reason}"
        )

        # Determine final status
        all_errors = lit_review_result.get("errors", [])
        if final_review_v2 and not all_errors:
            status = "success"
        elif final_review_v2 and all_errors:
            status = "partial"
        else:
            status = "failed"

        completed_at = datetime.utcnow()

        # Save full state for tests (in dev/test mode)
        save_workflow_state(
            workflow_name="supervised_lit_review",
            run_id=run_id,
            state={
                "input": dict(input_data) if hasattr(input_data, "_asdict") else input_data,
                "base_run_id": run_id,
                "final_review_v2": final_review_v2,
                "review_loop1": supervision_result.get("review_loop1"),
                "review_loop2": supervision_result.get("review_loop2"),
                "review_loop3": supervision_result.get("review_loop3"),
                "review_loop4": supervision_result.get("review_loop4"),
                "supervision": {
                    "loops_run": loops_run,
                    "completion_reason": completion_reason,
                    "loop_progress": supervision_result.get("loop_progress"),
                    "new_papers_added": new_paper_count,
                },
                "human_review_items": human_review_items,
                "paper_corpus": merged_corpus,
                "paper_summaries": merged_summaries,
                "clusters": clusters,
                "started_at": lit_review_result.get("started_at"),
                "completed_at": completed_at,
            },
        )

        return {
            "final_report": final_review_v2,
            "status": status,
            "langsmith_run_id": run_id,
            "errors": all_errors,
            "source_count": len(merged_corpus),
            "started_at": lit_review_result.get("started_at"),
            "completed_at": completed_at,
        }

    except Exception as e:
        logger.error(f"Supervision failed: {e}")
        return {
            "final_report": lit_review_result.get("final_report", final_review),
            "status": "partial",
            "langsmith_run_id": run_id,
            "errors": lit_review_result.get("errors", []) + [{"phase": "supervision", "error": str(e)}],
            "source_count": lit_review_result.get("source_count", 0),
            "started_at": lit_review_result.get("started_at"),
            "completed_at": datetime.utcnow(),
        }
