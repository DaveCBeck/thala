"""Content planning node using structured output."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_structured_output

from ..schemas import PlanningOutput
from ..prompts import EDITORIAL_STANCE_SECTION, PLANNING_SYSTEM_PROMPT, PLANNING_USER_TEMPLATE
from ..state import DeepDiveAssignment, EveningReadsState

logger = logging.getLogger(__name__)


async def plan_content_node(state: EveningReadsState) -> dict[str, Any]:
    """Plan the 4-part series using structured output.

    Uses OPUS for critical decision-making about topic division.

    Returns:
        State update with deep_dive_assignments and overview_scope
    """
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")
    citation_keys = state.get("extracted_citation_keys", [])

    # Format citation keys for the prompt
    citation_keys_str = ", ".join(citation_keys) if citation_keys else "None found"

    user_prompt = PLANNING_USER_TEMPLATE.format(
        literature_review=lit_review,
        citation_keys=citation_keys_str,
    )

    # Build system prompt with optional editorial stance
    system_prompt = PLANNING_SYSTEM_PROMPT
    if editorial_stance:
        system_prompt += EDITORIAL_STANCE_SECTION.format(editorial_stance=editorial_stance)
        logger.info("Planning with editorial stance context")

    logger.info(f"Planning series with {len(citation_keys)} available citation keys")

    try:
        planning_result: PlanningOutput = await get_structured_output(
            output_schema=PlanningOutput,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tier=ModelTier.OPUS,
            max_tokens=4096,
        )

        # Convert to state format
        deep_dive_assignments: list[DeepDiveAssignment] = []
        for dd in planning_result.deep_dives:
            # Validate anchor keys exist in our citation list
            valid_anchors = [k for k in dd.anchor_keys if k in citation_keys]
            if len(valid_anchors) < len(dd.anchor_keys):
                invalid = set(dd.anchor_keys) - set(valid_anchors)
                logger.warning(
                    f"Deep-dive {dd.id} has invalid anchor keys: {invalid}. "
                    f"Using only valid keys: {valid_anchors}"
                )

            deep_dive_assignments.append(
                DeepDiveAssignment(
                    id=dd.id,
                    title=dd.title,
                    theme=dd.theme,
                    structural_approach=dd.structural_approach,
                    anchor_keys=valid_anchors if valid_anchors else dd.anchor_keys[:3],
                    relevant_sections=dd.relevant_sections,
                )
            )

        logger.info(
            f"Planned series: {[(dd['title'], dd['structural_approach']) for dd in deep_dive_assignments]}"
        )

        return {
            "deep_dive_assignments": deep_dive_assignments,
            "overview_scope": planning_result.overview_scope,
        }

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {
            "errors": [{"node": "plan_content", "error": str(e)}],
            "status": "failed",
        }
