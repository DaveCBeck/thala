"""Supervisor analysis node for reviewing literature review theoretical depth."""

import logging
from typing import Any

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.wrappers.supervised_lit_review.supervision.types import (
    SupervisorDecision,
)
from workflows.wrappers.supervised_lit_review.supervision.prompts import (
    SUPERVISOR_SYSTEM,
    SUPERVISOR_USER,
)

logger = logging.getLogger(__name__)


async def analyze_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Analyze the literature review for theoretical gaps.

    Uses Opus with extended thinking to carefully assess whether the
    review has adequate theoretical grounding.

    Args:
        state: Current supervision state containing:
            - current_review: The literature review text to analyze
            - topic: Research topic
            - research_questions: List of research questions
            - source_count: Number of sources in corpus
            - issues_explored: Previously identified issues
            - iteration/max_iterations: Progress tracking

    Returns:
        State updates including the supervisor decision
    """
    current_review = state.get("current_review", "")
    issues_explored = state.get("issues_explored", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    topic = state.get("topic", "")
    research_questions = state.get("research_questions", [])
    source_count = state.get("source_count", 0)

    if not current_review:
        logger.warning("No review content to analyze")
        return {
            "decision": {
                "action": "error",
                "issue": None,
                "reasoning": "No review content to analyze",
            },
            "is_complete": False,
            "loop_error": {
                "loop_number": 1,
                "iteration": iteration,
                "node_name": "analyze_review",
                "error_type": "validation_error",
                "error_message": "No review content to analyze",
                "recoverable": False,
            },
        }

    rq_formatted = (
        "\n".join(f"- {q}" for q in research_questions)
        if research_questions
        else "None specified"
    )

    if issues_explored:
        explored_formatted = "\n".join(f"- {issue}" for issue in issues_explored)
    else:
        explored_formatted = "None yet"

    user_prompt = SUPERVISOR_USER.format(
        final_review=current_review,
        topic=topic,
        research_questions=rq_formatted,
        issues_explored=explored_formatted,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        source_count=source_count,
    )

    try:
        decision = await get_structured_output(
            output_schema=SupervisorDecision,
            user_prompt=user_prompt,
            system_prompt=SUPERVISOR_SYSTEM,
            tier=ModelTier.OPUS,
            thinking_budget=8000,
            max_tokens=12096,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.info(
            f"Supervisor decision: action={decision.action}, "
            f"reasoning={decision.reasoning[:100]}..."
        )

        updates: dict[str, Any] = {
            "decision": decision.model_dump(),
        }

        if decision.action == "pass_through":
            updates["is_complete"] = True
        elif decision.issue:
            updates["issues_explored"] = issues_explored + [decision.issue.topic]

        return updates

    except Exception as e:
        logger.error(f"Supervisor analysis failed: {e}")
        return {
            "decision": {
                "action": "error",
                "issue": None,
                "reasoning": f"Analysis failed: {e}",
            },
            "is_complete": False,
            "loop_error": {
                "loop_number": 1,
                "iteration": iteration,
                "node_name": "analyze_review",
                "error_type": "analysis_error",
                "error_message": str(e),
                "recoverable": True,
            },
        }
