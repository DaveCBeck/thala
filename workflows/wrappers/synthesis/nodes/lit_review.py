"""Phase 1: Academic literature review node."""

import logging
from typing import Any

from workflows.research.academic_lit_review import academic_lit_review
from workflows.shared.workflow_state_store import load_workflow_state
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
            workflow_name = "multi_lang"
        else:
            # Direct single-language call
            result = await academic_lit_review(
                topic=topic,
                research_questions=research_questions,
                quality=quality,
                language=language,
            )
            workflow_name = "academic_lit_review"

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

        # Load full state from workflow state store to get detailed results
        paper_corpus = {}
        paper_summaries = {}
        zotero_keys = {}

        run_id = result.get("langsmith_run_id")
        if run_id:
            full_state = load_workflow_state(workflow_name, run_id)
            if full_state:
                paper_corpus = full_state.get("paper_corpus", {})
                paper_summaries = full_state.get("paper_summaries", {})
                zotero_keys = full_state.get("zotero_keys", {})
                logger.info(
                    f"Loaded state: {len(paper_corpus)} papers, "
                    f"{len(paper_summaries)} summaries, {len(zotero_keys)} zotero keys"
                )
            else:
                logger.debug(f"No persisted state found for {workflow_name}/{run_id}")

        logger.info(
            f"Phase 1 complete: {result.get('source_count', 0)} papers analyzed, "
            f"status={result.get('status')}"
        )

        return {
            "lit_review_result": result,
            "paper_corpus": paper_corpus,
            "paper_summaries": paper_summaries,
            "zotero_keys": zotero_keys,
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
