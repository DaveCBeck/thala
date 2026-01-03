"""Supervisor analysis node for reviewing literature review theoretical depth."""

import logging
from typing import Any

from workflows.shared.llm_utils.models import ModelTier, get_llm
from workflows.research.subgraphs.academic_lit_review.supervision.types import (
    SupervisorDecision,
)
from workflows.research.subgraphs.academic_lit_review.supervision.prompts import (
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
            - input: Original input with topic and research questions
            - clusters: Thematic clusters for context
            - paper_corpus: Paper metadata for context
            - issues_explored: Previously identified issues
            - iteration/max_iterations: Progress tracking

    Returns:
        State updates including the supervisor decision
    """
    current_review = state.get("current_review", "")
    input_data = state.get("input", {})
    clusters = state.get("clusters", [])
    paper_corpus = state.get("paper_corpus", {})
    issues_explored = state.get("issues_explored", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    topic = input_data.get("topic", "")
    research_questions = input_data.get("research_questions", [])

    if not current_review:
        logger.warning("No review content to analyze")
        return {
            "decision": {"action": "pass_through", "reasoning": "No content to analyze"},
            "is_complete": True,
        }

    # Build cluster summary for context
    cluster_summary = _format_cluster_summary(clusters)

    # Format research questions
    rq_formatted = "\n".join(f"- {q}" for q in research_questions) if research_questions else "None specified"

    # Format previously explored issues
    if issues_explored:
        explored_formatted = "\n".join(f"- {issue}" for issue in issues_explored)
    else:
        explored_formatted = "None yet"

    # Build prompts
    user_prompt = SUPERVISOR_USER.format(
        final_review=current_review,
        topic=topic,
        research_questions=rq_formatted,
        cluster_summary=cluster_summary,
        issues_explored=explored_formatted,
        iteration=iteration + 1,  # Human-readable (1-indexed)
        max_iterations=max_iterations,
        corpus_size=len(paper_corpus),
    )

    # Use Opus with extended thinking for deep analysis
    llm = get_llm(
        tier=ModelTier.OPUS,
        thinking_budget=8000,
        max_tokens=12096,
    )

    try:
        # Get structured output
        structured_llm = llm.with_structured_output(SupervisorDecision)
        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        decision: SupervisorDecision = await structured_llm.ainvoke(messages)

        logger.info(
            f"Supervisor decision: action={decision.action}, "
            f"reasoning={decision.reasoning[:100]}..."
        )

        # Build state updates
        updates: dict[str, Any] = {
            "decision": decision.model_dump(),
        }

        if decision.action == "pass_through":
            updates["is_complete"] = True
        elif decision.issue:
            # Add issue to explored list to prevent re-exploration
            updates["issues_explored"] = issues_explored + [decision.issue.topic]

        return updates

    except Exception as e:
        logger.error(f"Supervisor analysis failed: {e}")
        # On error, treat as pass-through to avoid infinite loop
        return {
            "decision": {
                "action": "pass_through",
                "reasoning": f"Analysis error, defaulting to pass-through: {e}",
            },
            "is_complete": True,
        }


def _format_cluster_summary(clusters: list[dict]) -> str:
    """Format thematic clusters for supervisor context."""
    if not clusters:
        return "No thematic clusters available"

    lines = []
    for i, cluster in enumerate(clusters, 1):
        label = cluster.get("label", f"Cluster {i}")
        description = cluster.get("description", "")
        paper_count = len(cluster.get("paper_dois", []))
        lines.append(f"{i}. {label} ({paper_count} papers)")
        if description:
            lines.append(f"   {description[:150]}...")

    return "\n".join(lines)
