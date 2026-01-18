"""Phase 1: Academic literature review node."""

import logging
from typing import Any

from workflows.research.academic_lit_review import academic_lit_review
from workflows.wrappers.multi_lang.graph.api import multi_lang_research

logger = logging.getLogger(__name__)


async def run_lit_review(state: dict) -> dict[str, Any]:
    """Run academic literature review as first phase.

    Calls the academic_lit_review workflow to generate an initial
    literature review that will be enhanced in subsequent phases.
    If multi_lang_config is provided, uses multi_lang_research instead.
    """
    input_data = state.get("input", {})

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])
    quality = input_data.get("quality", "standard")
    language = input_data.get("language_code", "en")
    multi_lang_config = input_data.get("multi_lang_config")

    logger.info(f"Phase 1: Starting academic literature review for '{topic}'")

    try:
        if multi_lang_config is not None:
            # Use multi-language research wrapper
            logger.info("Multi-lang mode enabled for academic lit review")
            result_obj = await multi_lang_research(
                topic=topic,
                mode=multi_lang_config.get("mode", "set_languages"),
                languages=multi_lang_config.get("languages"),
                research_questions=research_questions,
                workflow="academic",
                quality=quality,
            )
            result = result_obj.to_dict()
        else:
            # Direct single-language call
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
