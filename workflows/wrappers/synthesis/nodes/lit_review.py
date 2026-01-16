"""Phase 1: Academic literature review node."""

import logging
from typing import Any

from workflows.research.academic_lit_review import academic_lit_review

logger = logging.getLogger(__name__)


async def run_lit_review(state: dict) -> dict[str, Any]:
    """Run academic literature review as first phase.

    Calls the academic_lit_review workflow to generate an initial
    literature review that will be enhanced in subsequent phases.
    """
    input_data = state.get("input", {})

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])
    quality = input_data.get("quality", "standard")
    language = input_data.get("language_code", "en")

    logger.info(f"Phase 1: Starting academic literature review for '{topic}'")

    try:
        result = await academic_lit_review(
            topic=topic,
            research_questions=research_questions,
            quality=quality,
            language=language,
        )

        if result.get("status") == "failed":
            logger.error(f"Literature review failed: {result.get('errors', [])}")
            return {
                "lit_review_result": result,
                "paper_corpus": {},
                "paper_summaries": {},
                "zotero_keys": {},
                "current_phase": "supervision",
                "errors": [{"phase": "lit_review", "error": "Literature review failed"}],
            }

        logger.info(
            f"Phase 1 complete: {result.get('source_count', 0)} papers analyzed, "
            f"status={result.get('status')}"
        )

        return {
            "lit_review_result": result,
            # Note: paper_corpus, paper_summaries, zotero_keys are populated
            # from the workflow state store if needed by downstream phases
            "current_phase": "supervision",
        }

    except Exception as e:
        logger.error(f"Literature review failed with exception: {e}")
        return {
            "lit_review_result": None,
            "paper_corpus": {},
            "paper_summaries": {},
            "zotero_keys": {},
            "current_phase": "supervision",
            "errors": [{"phase": "lit_review", "error": str(e)}],
        }
