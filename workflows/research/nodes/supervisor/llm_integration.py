"""LLM integration for structured supervisor decisions."""

import logging

from workflows.research.state import SupervisorDecision
from workflows.shared.llm_utils import get_structured_output, ModelTier

logger = logging.getLogger(__name__)


async def _get_supervisor_decision_structured(
    llm, system_prompt: str, user_prompt: str, brief: dict
) -> tuple[str, dict, SupervisorDecision]:
    """Try to get supervisor decision using structured output.

    Returns: (action, action_data, decision) tuple

    Raises: Exception if structured output fails
    """
    decision: SupervisorDecision = await get_structured_output(
        output_schema=SupervisorDecision,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        tier=ModelTier.OPUS,
    )

    action = decision.action
    action_data = {}

    if action == "conduct_research":
        # Convert research_questions to the expected format
        action_data["questions"] = [
            {"question": q, "context": brief.get("topic", "")}
            for q in decision.research_questions
        ]
        # Include allocation from LLM decision
        action_data["llm_allocation"] = {
            "web_count": decision.web_researchers,
        }
    elif action == "refine_draft":
        action_data["updates"] = decision.draft_updates or ""
        action_data["gaps"] = decision.remaining_gaps

    logger.info(
        f"Supervisor (structured): {action}, reasoning: {decision.reasoning[:100]}..."
    )

    return action, action_data, decision
