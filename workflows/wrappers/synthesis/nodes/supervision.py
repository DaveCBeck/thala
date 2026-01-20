"""Phase 2: Supervision node for theoretical depth and literature expansion."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance import enhance_report

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="SynthesisSupervision")
async def run_supervision(state: dict) -> dict[str, Any]:
    """Run supervision workflow to enhance literature review.

    Runs the enhance.supervision workflow which includes:
    - Loop 1: Theoretical depth (identifies and fills theoretical gaps)
    - Loop 2: Literature expansion (discovers and integrates new literature)

    Note: This phase is skipped in "test" quality mode.
    """
    input_data = state.get("input", {})
    quality_settings = state.get("quality_settings", {})
    lit_review_result = state.get("lit_review_result", {})

    # Skip supervision if configured
    if quality_settings.get("skip_supervision", False):
        logger.info("Phase 2: Skipping supervision (test mode)")
        return {
            "supervision_result": None,
            "current_phase": "research_targets",
        }

    # Get the report from literature review
    report = lit_review_result.get("final_report", "")
    if not report:
        logger.warning("Phase 2: No report from literature review, skipping supervision")
        return {
            "supervision_result": None,
            "current_phase": "research_targets",
        }

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])
    quality = input_data.get("quality", "standard")

    # Get paper corpus from previous phase if available
    paper_corpus = state.get("paper_corpus", {})
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})

    logger.info(f"Phase 2: Starting supervision for '{topic}'")

    try:
        result = await enhance_report(
            report=report,
            topic=topic,
            research_questions=research_questions,
            quality=quality,
            loops="all",  # Run both supervision loops
            run_editing=False,  # Don't run editing here - we do it at the end
            paper_corpus=paper_corpus,
            paper_summaries=paper_summaries,
            zotero_keys=zotero_keys,
        )

        # Update paper corpus with newly discovered papers
        updated_paper_corpus = result.get("paper_corpus", paper_corpus)
        updated_paper_summaries = result.get("paper_summaries", paper_summaries)
        updated_zotero_keys = result.get("zotero_keys", zotero_keys)

        logger.info(
            f"Phase 2 complete: status={result.get('status')}, "
            f"papers_after={len(updated_paper_corpus)}"
        )

        return {
            "supervision_result": result,
            "paper_corpus": updated_paper_corpus,
            "paper_summaries": updated_paper_summaries,
            "zotero_keys": updated_zotero_keys,
            "current_phase": "research_targets",
        }

    except Exception as e:
        logger.error(f"Supervision failed with exception: {e}")
        return {
            "supervision_result": None,
            "current_phase": "research_targets",
            "errors": [{"phase": "supervision", "error": str(e)}],
        }
