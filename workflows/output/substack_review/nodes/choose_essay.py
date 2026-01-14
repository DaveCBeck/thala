"""Choosing agent node with structured output."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.structured import get_structured_output
from ..state import SubstackReviewState
from ..schemas import ChoosingAgentOutput
from ..prompts import CHOOSING_SYSTEM_PROMPT, CHOOSING_USER_TEMPLATE

logger = logging.getLogger(__name__)


async def choose_essay_node(state: SubstackReviewState) -> dict[str, Any]:
    """Select the best essay using structured output.

    Evaluates all three essays and returns the winning angle
    with detailed evaluation reasoning.
    """
    essay_drafts = state.get("essay_drafts", [])

    # Build essay lookup by angle
    essays_by_angle = {e["angle"]: e["content"] for e in essay_drafts}

    # Check we have all three essays
    expected_angles = {"puzzle", "finding", "contrarian"}
    missing = expected_angles - set(essays_by_angle.keys())

    if missing:
        logger.error(f"Missing essays for angles: {missing}")
        return {
            "status": "failed",
            "errors": [
                {"node": "choose_essay", "error": f"Missing essays: {missing}"}
            ],
        }

    user_prompt = CHOOSING_USER_TEMPLATE.format(
        essay_puzzle=essays_by_angle["puzzle"],
        essay_finding=essays_by_angle["finding"],
        essay_contrarian=essays_by_angle["contrarian"],
    )

    logger.info("Invoking choosing agent with OPUS")

    try:
        result: ChoosingAgentOutput = await get_structured_output(
            output_schema=ChoosingAgentOutput,
            user_prompt=user_prompt,
            system_prompt=CHOOSING_SYSTEM_PROMPT,
            tier=ModelTier.OPUS,
            max_tokens=4096,
        )

        logger.info(
            f"Selected essay: {result.selected} (close_call={result.close_call})"
        )

        return {
            "selected_angle": result.selected,
            "selection_reasoning": result.deciding_factors,
            "essay_evaluations": {
                angle: eval_obj.model_dump()
                for angle, eval_obj in result.evaluations.items()
            },
        }

    except Exception as e:
        logger.error(f"Choosing agent failed: {e}")
        return {
            "status": "failed",
            "errors": [{"node": "choose_essay", "error": str(e)}],
        }
