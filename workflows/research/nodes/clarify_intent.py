"""
Clarify intent node.

Determines if the user's research request needs clarification.
If ambiguous, generates clarifying questions.
"""

import json
import logging
from typing import Any

from workflows.research.state import DeepResearchState, ClarificationQuestion
from workflows.research.prompts import CLARIFY_INTENT_SYSTEM, CLARIFY_INTENT_HUMAN, get_today_str
from workflows.research.prompts.translator import get_translated_prompt
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


async def clarify_intent(state: DeepResearchState) -> dict[str, Any]:
    """Determine if clarification is needed for the research request.

    Returns:
        - clarification_needed: bool
        - clarification_questions: list[ClarificationQuestion] if needed
        - current_status: updated status
    """
    query = state["input"]["query"]
    language_config = state.get("primary_language_config")

    # Get language-appropriate prompts
    if language_config and language_config["code"] != "en":
        system_prompt_template = await get_translated_prompt(
            CLARIFY_INTENT_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="clarify_intent_system",
        )
        human_prompt_template = await get_translated_prompt(
            CLARIFY_INTENT_HUMAN,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="clarify_intent_human",
        )
    else:
        system_prompt_template = CLARIFY_INTENT_SYSTEM
        human_prompt_template = CLARIFY_INTENT_HUMAN

    system_prompt = system_prompt_template.format(date=get_today_str())
    human_prompt = human_prompt_template.format(query=query)

    llm = get_llm(ModelTier.HAIKU)  # Fast model for simple decision

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ])

        content = response.content.strip()

        # Extract JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)

        if result.get("need_clarification", False):
            questions = [
                ClarificationQuestion(
                    question=q.get("question", ""),
                    options=q.get("options"),
                )
                for q in result.get("questions", [])
            ]

            logger.info(f"Clarification needed: {len(questions)} questions")

            return {
                "clarification_needed": True,
                "clarification_questions": questions,
                "current_status": "awaiting_clarification",
            }
        else:
            logger.info("No clarification needed, proceeding with research")

            return {
                "clarification_needed": False,
                "clarification_questions": [],
                "current_status": "creating_brief",
            }

    except Exception as e:
        logger.error(f"Clarify intent failed: {e}")

        # Proceed without clarification on error
        return {
            "clarification_needed": False,
            "clarification_questions": [],
            "errors": [{"node": "clarify_intent", "error": str(e)}],
            "current_status": "creating_brief",
        }
