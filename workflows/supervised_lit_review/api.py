"""Main entry point for supervised academic literature review workflow.

This wrapper runs the academic_lit_review workflow first, then applies
configurable supervision loops to enhance quality.
"""

import logging
from datetime import datetime
from typing import Any, Literal, Optional

from workflows.academic_lit_review.graph.api import academic_lit_review
from workflows.academic_lit_review.state import QUALITY_PRESETS
from workflows.supervised_lit_review.supervision.orchestration import (
    run_supervision_configurable,
)

logger = logging.getLogger(__name__)


async def supervised_lit_review(
    topic: str,
    research_questions: list[str],
    quality: Literal["test", "quick", "standard", "comprehensive", "high_quality"] = "standard",
    language: str = "en",
    date_range: Optional[tuple[int, int]] = None,
    include_books: bool = True,
    focus_areas: Optional[list[str]] = None,
    exclude_terms: Optional[list[str]] = None,
    max_papers: Optional[int] = None,
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
        include_books: Whether to include book sources (default: True)
        focus_areas: Optional specific areas to prioritize
        exclude_terms: Optional terms to filter out
        max_papers: Override default max papers for quality tier
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
    logger.info(f"Starting supervised literature review: {topic}")
    logger.info(f"Supervision loops: {supervision_loops}")

    # Step 1: Run the core literature review (without supervision)
    lit_review_result = await academic_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        language=language,
        date_range=date_range,
        include_books=include_books,
        focus_areas=focus_areas,
        exclude_terms=exclude_terms,
        max_papers=max_papers,
    )

    # Check for errors in lit review
    has_errors = bool(lit_review_result.get("errors"))
    if has_errors:
        logger.warning(f"Lit review had errors: {lit_review_result['errors']}")

    final_review = lit_review_result.get("final_review", "")

    # A valid review should be substantial (at least 500 chars)
    # If we have errors and a tiny "review", it's not a real review
    has_valid_review = final_review and len(final_review) >= 500

    # If no supervision requested, no valid review, or lit review failed - return as-is
    if supervision_loops == "none" or not has_valid_review:
        if has_errors and not has_valid_review:
            logger.error("Lit review failed - skipping supervision loops")
        logger.info("Skipping supervision (disabled or no review content)")
        # Determine status: failed if no valid review, otherwise use lit_review status
        if has_errors and not has_valid_review:
            status = "failed"
        else:
            status = lit_review_result.get("status", "success" if has_valid_review else "failed")
        return {
            **lit_review_result,
            # Ensure standardized fields from academic_lit_review are present
            "final_report": lit_review_result.get("final_report", final_review),
            "status": status,
            "final_review_v2": None,
            "supervision": None,
            "human_review_items": [],
        }

    # Step 2: Apply supervision loops
    logger.info(f"Applying supervision loops to {len(final_review)} char review")

    # Get quality settings for supervision
    quality_preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
    max_stages = quality_preset.get("max_stages", 3)

    # Build input data for supervision
    from workflows.academic_lit_review.state import LitReviewInput

    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        include_books=include_books,
        focus_areas=focus_areas,
        exclude_terms=exclude_terms,
        max_papers=max_papers,
        language_code=language,
    )

    try:
        supervision_result = await run_supervision_configurable(
            review=final_review,
            paper_corpus=lit_review_result.get("paper_corpus", {}),
            paper_summaries=lit_review_result.get("paper_summaries", {}),
            zotero_keys=lit_review_result.get("zotero_keys", {}),
            clusters=lit_review_result.get("clusters", []),
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
        original_corpus = lit_review_result.get("paper_corpus", {})
        original_summaries = lit_review_result.get("paper_summaries", {})

        merged_corpus = {**original_corpus, **added_papers}
        merged_summaries = {**original_summaries, **added_summaries}

        new_paper_count = len(added_papers) - len(
            set(added_papers.keys()) & set(original_corpus.keys())
        )

        logger.info(
            f"Supervision complete: loops={loops_run}, "
            f"{new_paper_count} new papers, {len(human_review_items)} items for review. "
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

        return {
            **lit_review_result,
            "final_review_v2": final_review_v2,
            "final_report": final_review_v2,  # Standardized: use supervised version
            "status": status,  # Standardized status
            "paper_corpus": merged_corpus,
            "paper_summaries": merged_summaries,
            "supervision": {
                "loops_run": loops_run,
                "completion_reason": completion_reason,
                "loop_progress": supervision_result.get("loop_progress"),
                "new_papers_added": new_paper_count,
            },
            "human_review_items": human_review_items,
            "completed_at": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Supervision failed: {e}")
        return {
            **lit_review_result,
            "final_report": lit_review_result.get("final_report", final_review),  # Use original
            "status": "partial",  # Lit review succeeded but supervision failed
            "final_review_v2": None,
            "supervision": {"error": str(e)},
            "human_review_items": [],
            "errors": lit_review_result.get("errors", [])
            + [{"phase": "supervision", "error": str(e)}],
        }
