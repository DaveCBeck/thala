"""Loop 2 integration nodes for literature base expansion and findings integration."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_llm
from workflows.research.academic_lit_review.graph.api import academic_lit_review

from ...types import LiteratureBase
from workflows.research.academic_lit_review.quality_presets import QualitySettings
from ...prompts import LOOP2_INTEGRATOR_SYSTEM, LOOP2_INTEGRATOR_USER

logger = logging.getLogger(__name__)


async def run_academic_review_for_base(
    literature_base: LiteratureBase,
    parent_topic: str,
    quality_settings: QualitySettings,
) -> dict[str, Any]:
    """Run academic_lit_review workflow for a literature base expansion.

    This replaces the previous mini_review with the more robust academic_lit_review
    workflow which has proper Zotero key verification built in.

    Papers go directly to ES/Zotero via the nested workflow - no need to
    filter or return corpus data.
    """
    # Determine quality preset based on parent settings
    quality_preset = "quick"
    if quality_settings.get("max_papers", 100) <= 10:  # test preset
        quality_preset = "test"

    logger.info(
        f"Running academic_lit_review for base '{literature_base.name}' "
        f"with quality='{quality_preset}'"
    )

    result = await academic_lit_review(
        topic=f"{parent_topic} - {literature_base.name}",
        research_questions=literature_base.search_queries,
        quality=quality_preset,
    )

    papers_processed = len(result.get("paper_corpus", {}))
    logger.info(f"Academic review complete: {papers_processed} papers processed")

    # Return only what's needed for integration (the mini-review text and citation keys)
    return {
        "mini_review_text": result.get("final_review", ""),
        "zotero_keys": result.get("zotero_keys", {}),
    }


async def run_mini_review_node(state: dict) -> dict:
    """Execute mini-review on identified literature base."""
    decision = state.get("decision")
    errors = state.get("errors", [])

    if not decision:
        logger.warning("run_mini_review_node called without decision")
        return {
            "mini_review_failed": True,
            "errors": errors + [{
                "loop_number": 2,
                "iteration": state.get("iteration", 0),
                "node_name": "run_mini_review",
                "error_type": "validation_error",
                "error_message": "No decision provided",
                "recoverable": False,
            }],
        }

    if decision["action"] != "expand_base":
        logger.warning(f"run_mini_review_node called with action: {decision['action']}")
        return {
            "mini_review_failed": True,
            "errors": errors + [{
                "loop_number": 2,
                "iteration": state.get("iteration", 0),
                "node_name": "run_mini_review",
                "error_type": "validation_error",
                "error_message": f"Expected expand_base action, got: {decision['action']}",
                "recoverable": False,
            }],
        }

    literature_base = LiteratureBase(**decision["literature_base"])
    logger.info(f"Running mini-review for: {literature_base.name}")

    parent_topic = state["topic"]

    mini_review_result = await run_academic_review_for_base(
        literature_base=literature_base,
        parent_topic=parent_topic,
        quality_settings=state["quality_settings"],
    )

    mini_review_text = mini_review_result.get("mini_review_text", "")
    new_zotero_keys = mini_review_result.get("zotero_keys", {})

    logger.info(
        f"Mini-review complete: {len(new_zotero_keys)} papers, "
        f"{len(mini_review_text)} chars"
    )

    return {
        "decision": {
            **decision,
            "mini_review_text": mini_review_text,
            "new_zotero_keys": new_zotero_keys,
        }
    }


async def integrate_findings_node(state: dict) -> dict:
    """Integrate mini-review findings into main review."""
    iteration = state.get("iteration", 0)
    errors = state.get("errors", [])
    iterations_failed = state.get("iterations_failed", 0)

    try:
        decision = state.get("decision")
        if not decision or not decision.get("literature_base"):
            logger.error("integrate_findings_node called without valid decision")
            return {
                "errors": errors + [{
                    "loop_number": 2,
                    "iteration": iteration,
                    "node_name": "integrate_findings",
                    "error_type": "validation_error",
                    "error_message": "No valid decision with literature_base",
                    "recoverable": False,
                }],
                "iterations_failed": iterations_failed + 1,
                "integration_failed": True,
            }

        literature_base = LiteratureBase(**decision["literature_base"])

        logger.info(f"Integrating findings for: {literature_base.name}")

        mini_review_text = decision.get("mini_review_text", "")
        new_zotero_keys = decision.get("new_zotero_keys", {})

        if not mini_review_text or len(mini_review_text.strip()) < 100:
            logger.error(
                f"Mini-review too short for base: {literature_base.name} "
                f"({len(mini_review_text.strip()) if mini_review_text else 0} chars)"
            )
            return {
                "errors": errors + [{
                    "loop_number": 2,
                    "iteration": iteration,
                    "node_name": "integrate_findings",
                    "error_type": "validation_error",
                    "error_message": f"Mini-review failed or too short ({len(mini_review_text.strip()) if mini_review_text else 0} chars)",
                    "recoverable": True,
                }],
                "iterations_failed": iterations_failed + 1,
                "integration_failed": True,
            }

        # Format citation keys for the prompt (keys only, no titles since we don't have summaries)
        citation_keys = "\n".join(f"[@{key}]" for key in new_zotero_keys.values())

        user_prompt = LOOP2_INTEGRATOR_USER.format(
            current_review=state["current_review"],
            literature_base_name=literature_base.name,
            mini_review=mini_review_text,
            integration_strategy=literature_base.integration_strategy,
            new_citation_keys=citation_keys or "None",
        )

        llm = get_llm(ModelTier.OPUS, max_tokens=16384)
        response = await llm.ainvoke([
            {"role": "system", "content": LOOP2_INTEGRATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])

        updated_review = response.content
        explored_bases = state.get("explored_bases", []) + [literature_base.name]

        logger.info(
            f"Integration complete: review length {len(updated_review)}, "
            f"{len(new_zotero_keys)} new citations available"
        )

        return {
            "current_review": updated_review,
            "explored_bases": explored_bases,
            "iteration": iteration + 1,
        }

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return {
            "errors": errors + [{
                "loop_number": 2,
                "iteration": iteration,
                "node_name": "integrate_findings",
                "error_type": "integration_error",
                "error_message": str(e),
                "recoverable": True,
            }],
            "iterations_failed": iterations_failed + 1,
            "integration_failed": True,
        }
