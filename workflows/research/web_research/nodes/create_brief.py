"""
Create brief node.

Transforms the user's query (and any clarifications) into a structured research brief.
"""

import json
import logging
from typing import Any

from workflows.research.web_research.state import DeepResearchState, ResearchBrief
from workflows.research.web_research.prompts import (
    CREATE_BRIEF_SYSTEM,
    CREATE_BRIEF_HUMAN,
    get_today_str,
)
from workflows.research.web_research.utils import (
    load_prompts_with_translation,
    extract_json_from_llm_response,
)
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)


async def create_brief(state: DeepResearchState) -> dict[str, Any]:
    """Create a structured research brief from the query.

    Returns:
        - research_brief: ResearchBrief
        - current_status: updated status
    """
    query = state["input"]["query"]
    clarifications = state.get("clarification_responses") or {}
    language_config = state.get("primary_language_config")

    system_prompt_template, human_prompt_template = await load_prompts_with_translation(
        CREATE_BRIEF_SYSTEM,
        CREATE_BRIEF_HUMAN,
        language_config,
        "create_brief_system",
        "create_brief_human",
    )

    system_prompt = system_prompt_template.format(date=get_today_str())
    human_prompt = human_prompt_template.format(
        query=query,
        clarifications=json.dumps(clarifications) if clarifications else "None",
    )

    llm = get_llm(ModelTier.SONNET)

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]
        )

        brief_data = extract_json_from_llm_response(response.content)

        brief = ResearchBrief(
            topic=brief_data.get("topic", query),
            objectives=brief_data.get("objectives", []),
            scope=brief_data.get("scope", ""),
            key_questions=brief_data.get("key_questions", []),
            memory_context="",
        )

        logger.debug(
            f"Created research brief: topic='{brief['topic'][:50]}...', "
            f"{len(brief['objectives'])} objectives, {len(brief['key_questions'])} questions"
        )

        return {
            "research_brief": brief,
            "current_status": "searching_memory",
        }

    except Exception as e:
        logger.error(f"Create brief failed: {e}")

        # Fallback brief
        brief = ResearchBrief(
            topic=query,
            objectives=[f"Research: {query}"],
            scope="General research",
            key_questions=[query],
            memory_context="",
        )

        return {
            "research_brief": brief,
            "errors": [{"node": "create_brief", "error": str(e)}],
            "current_status": "searching_memory",
        }
