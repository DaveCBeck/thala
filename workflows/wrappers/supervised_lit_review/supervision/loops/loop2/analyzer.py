"""Loop 2 analysis node for identifying missing literature bases."""

import logging

from workflows.shared.llm_utils import ModelTier, get_structured_output

from ...types import LiteratureBaseDecision
from ...prompts import LOOP2_ANALYZER_SYSTEM, LOOP2_ANALYZER_USER

logger = logging.getLogger(__name__)


async def analyze_for_bases_node(state: dict) -> dict:
    """Analyze review to identify missing literature base."""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    logger.info(
        f"Loop 2 iteration {iteration + 1}/{max_iterations}: "
        "Analyzing for missing literature bases"
    )

    topic = state["topic"]
    research_questions = state["research_questions"]
    explored_bases_text = "\n".join(
        f"- {base}" for base in state.get("explored_bases", [])
    ) or "None yet"

    user_prompt = LOOP2_ANALYZER_USER.format(
        review=state["current_review"],
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
        explored_bases=explored_bases_text,
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        response = await get_structured_output(
            output_schema=LiteratureBaseDecision,
            user_prompt=user_prompt,
            system_prompt=LOOP2_ANALYZER_SYSTEM,
            tier=ModelTier.OPUS,
            max_tokens=2048,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.info(f"Analyzer decision: {response.action}")
        if response.action == "expand_base":
            logger.info(f"Identified literature base: {response.literature_base.name}")

        return {"decision": response.model_dump()}

    except Exception as e:
        logger.error(f"Loop 2 analysis failed: {e}")
        errors = state.get("errors", [])
        return {
            "decision": {
                "action": "error",
                "literature_base": None,
                "reasoning": f"Analysis failed: {e}",
            },
            "errors": errors + [{
                "loop_number": 2,
                "iteration": iteration,
                "node_name": "analyze_for_bases",
                "error_type": "analysis_error",
                "error_message": str(e),
                "recoverable": True,
            }],
        }
