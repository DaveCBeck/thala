"""
Iterate plan node.

THALA-SPECIFIC: Customizes the research plan based on the user's existing
knowledge, beliefs, and preferences found in memory stores.

Uses OPUS for deep understanding of user's knowledge landscape.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from workflows.research.web_research.state import DeepResearchState
from workflows.research.web_research.prompts import (
    ITERATE_PLAN_SYSTEM,
    ITERATE_PLAN_HUMAN,
    get_today_str,
)
from workflows.research.web_research.utils import load_prompts_with_translation
from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


class IteratePlanResponse(BaseModel):
    """Structured output for customized research plan."""

    user_knows: list[str] = Field(
        default_factory=list, description="What the user already understands well"
    )
    knowledge_gaps: list[str] = Field(
        default_factory=list, description="Specific gaps to fill with research"
    )
    priority_questions: list[str] = Field(
        default_factory=list, description="Prioritized questions based on gaps"
    )
    avoid_researching: list[str] = Field(
        default_factory=list, description="Topics the user already knows well"
    )
    potential_challenges: list[str] = Field(
        default_factory=list,
        description="Areas where findings might challenge existing beliefs",
    )
    research_strategy: str = Field(
        default="", description="Overall approach given their existing knowledge"
    )


async def iterate_plan(state: DeepResearchState) -> dict[str, Any]:
    """Customize research plan based on user's existing knowledge.

    This node uses OPUS to deeply understand:
    - What the user already knows
    - Knowledge gaps to fill
    - Their existing beliefs and preferences
    - Areas where new research might challenge existing views

    Returns:
        - research_plan: Customized plan string
        - pending_questions: Initial questions based on gaps
        - current_status: updated status
    """
    brief = state.get("research_brief")
    memory_context = state.get("memory_context", "")
    language_config = state.get("primary_language_config")

    if not brief:
        logger.warning("No research brief available for plan iteration")
        return {
            "research_plan": "No research brief - using default approach.",
            "current_status": "supervising",
        }

    system_prompt_template, human_prompt_template = await load_prompts_with_translation(
        ITERATE_PLAN_SYSTEM,
        ITERATE_PLAN_HUMAN,
        language_config,
        "iterate_plan_system",
        "iterate_plan_human",
    )

    system_prompt = system_prompt_template.format(
        date=get_today_str(),
        memory_context=memory_context or "No existing knowledge found in memory.",
        research_brief=json.dumps(brief, indent=2),
    )

    try:
        result: IteratePlanResponse = await get_structured_output(
            output_schema=IteratePlanResponse,
            user_prompt=human_prompt_template,
            system_prompt=system_prompt,
            tier=ModelTier.OPUS,
        )

        # Build structured plan from response
        plan_parts = []

        if result.user_knows:
            plan_parts.append("**What you already know:**")
            for item in result.user_knows:
                plan_parts.append(f"- {item}")

        if result.knowledge_gaps:
            plan_parts.append("\n**Knowledge gaps to fill:**")
            for item in result.knowledge_gaps:
                plan_parts.append(f"- {item}")

        if result.avoid_researching:
            plan_parts.append("\n**Areas to skip (already well understood):**")
            for item in result.avoid_researching:
                plan_parts.append(f"- {item}")

        if result.potential_challenges:
            plan_parts.append("\n**Potential challenges to existing beliefs:**")
            for item in result.potential_challenges:
                plan_parts.append(f"- {item}")

        if result.research_strategy:
            plan_parts.append(f"\n**Strategy:** {result.research_strategy}")

        research_plan = "\n".join(plan_parts)

        logger.debug(
            f"Customized research plan: {len(result.knowledge_gaps)} gaps, "
            f"{len(result.priority_questions)} priority questions"
        )

        return {
            "research_plan": research_plan,
            "current_status": "supervising",
        }

    except Exception as e:
        logger.error(f"Iterate plan failed: {e}")

        # Fallback: default plan
        return {
            "research_plan": f"Default research approach for: {brief.get('topic', 'unknown topic')}",
            "errors": [{"node": "iterate_plan", "error": str(e)}],
            "current_status": "supervising",
        }
