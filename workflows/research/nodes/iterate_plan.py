"""
Iterate plan node.

THALA-SPECIFIC: Customizes the research plan based on the user's existing
knowledge, beliefs, and preferences found in memory stores.

Uses OPUS for deep understanding of user's knowledge landscape.
"""

import json
import logging
from typing import Any

from workflows.research.state import DeepResearchState
from workflows.research.prompts import ITERATE_PLAN_SYSTEM, ITERATE_PLAN_HUMAN, get_today_str
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


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

    if not brief:
        logger.warning("No research brief available for plan iteration")
        return {
            "research_plan": "No research brief - using default approach.",
            "current_status": "supervising",
        }

    system_prompt = ITERATE_PLAN_SYSTEM.format(
        date=get_today_str(),
        memory_context=memory_context or "No existing knowledge found in memory.",
        research_brief=json.dumps(brief, indent=2),
    )

    llm = get_llm(ModelTier.OPUS)  # OPUS for deep understanding

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ITERATE_PLAN_HUMAN},
        ])

        content = response.content.strip()

        # Extract JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        plan_data = json.loads(content)

        # Build structured plan
        plan_parts = []

        if plan_data.get("user_knows"):
            plan_parts.append("**What you already know:**")
            for item in plan_data["user_knows"]:
                plan_parts.append(f"- {item}")

        if plan_data.get("knowledge_gaps"):
            plan_parts.append("\n**Knowledge gaps to fill:**")
            for item in plan_data["knowledge_gaps"]:
                plan_parts.append(f"- {item}")

        if plan_data.get("avoid_researching"):
            plan_parts.append("\n**Areas to skip (already well understood):**")
            for item in plan_data["avoid_researching"]:
                plan_parts.append(f"- {item}")

        if plan_data.get("potential_challenges"):
            plan_parts.append("\n**Potential challenges to existing beliefs:**")
            for item in plan_data["potential_challenges"]:
                plan_parts.append(f"- {item}")

        if plan_data.get("research_strategy"):
            plan_parts.append(f"\n**Strategy:** {plan_data['research_strategy']}")

        research_plan = "\n".join(plan_parts)

        # Extract priority questions for initial research
        priority_questions = plan_data.get("priority_questions", [])

        logger.info(
            f"Customized research plan: {len(plan_data.get('knowledge_gaps', []))} gaps, "
            f"{len(priority_questions)} priority questions"
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
