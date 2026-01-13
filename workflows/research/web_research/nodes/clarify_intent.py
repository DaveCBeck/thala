"""
Clarify intent node.

Determines if the user's research request needs clarification.
If ambiguous, generates clarifying questions.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from workflows.research.web_research.state import DeepResearchState, ClarificationQuestion
from workflows.research.web_research.prompts import CLARIFY_INTENT_SYSTEM, CLARIFY_INTENT_HUMAN, get_today_str
from workflows.research.web_research.utils import load_prompts_with_translation
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


class ClarificationQuestionModel(BaseModel):
    """Single clarification question."""

    question: str
    options: list[str] | None = None


class ClarificationResponse(BaseModel):
    """Structured output for clarification decision."""

    need_clarification: bool = Field(
        description="Whether the request needs clarification"
    )
    questions: list[ClarificationQuestionModel] = Field(
        default_factory=list,
        description="List of clarifying questions if needed"
    )


async def clarify_intent(state: DeepResearchState) -> dict[str, Any]:
    """Determine if clarification is needed for the research request.

    Returns:
        - clarification_needed: bool
        - clarification_questions: list[ClarificationQuestion] if needed
        - current_status: updated status
    """
    query = state["input"]["query"]
    language_config = state.get("primary_language_config")

    system_prompt_template, human_prompt_template = await load_prompts_with_translation(
        CLARIFY_INTENT_SYSTEM,
        CLARIFY_INTENT_HUMAN,
        language_config,
        "clarify_intent_system",
        "clarify_intent_human",
    )

    system_prompt = system_prompt_template.format(date=get_today_str())
    human_prompt = human_prompt_template.format(query=query)

    try:
        result: ClarificationResponse = await get_structured_output(
            output_schema=ClarificationResponse,
            user_prompt=human_prompt,
            system_prompt=system_prompt,
            tier=ModelTier.HAIKU,
        )

        if result.need_clarification:
            questions = [
                ClarificationQuestion(
                    question=q.question,
                    options=q.options,
                )
                for q in result.questions
            ]

            logger.debug(f"Clarification needed: {len(questions)} questions generated")

            return {
                "clarification_needed": True,
                "clarification_questions": questions,
                "current_status": "awaiting_clarification",
            }
        else:
            logger.debug("No clarification needed")

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
