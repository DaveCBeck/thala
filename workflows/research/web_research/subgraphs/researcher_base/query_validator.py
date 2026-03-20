"""Query validation utilities."""

import logging

from workflows.research.web_research.state import QueryValidationBatch
from workflows.shared.llm_utils import ModelTier, invoke

logger = logging.getLogger(__name__)


async def validate_queries(
    queries: list[str],
    research_question: str,
    research_brief: dict | None = None,
    draft_notes: str | None = None,
) -> list[str]:
    """
    Validate queries using LLM to ensure they're relevant to the research.

    Args:
        queries: Generated search queries to validate
        research_question: The original research question
        research_brief: Optional research brief for context
        draft_notes: Optional current draft/notes for context

    Returns:
        List of validated queries that are relevant to the research
    """
    if not queries:
        return []

    # Build context
    context_parts = [f"Research Question: {research_question}"]
    if research_brief:
        context_parts.append(f"Topic: {research_brief.get('topic', '')}")
        if research_brief.get("objectives"):
            context_parts.append(f"Objectives: {', '.join(research_brief['objectives'][:3])}")
    if draft_notes:
        context_parts.append(f"Current Notes: {draft_notes[:500]}...")

    context = "\n".join(context_parts)
    queries_list = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))

    system_prompt = (
        "Validate whether search queries are relevant to a research task.\n"
        "For each query, determine if it's actually relevant to the research question.\n"
        "Reject queries that:\n"
        "- Contain system metadata (iteration counts, percentages, internal state)\n"
        "- Are about completely unrelated topics\n"
        "- Are too vague or generic to be useful\n\n"
        "Accept queries that would help find information about the research topic."
    )

    user_prompt = f"""{context}

Proposed Search Queries:
{queries_list}"""

    try:
        result: QueryValidationBatch = await invoke(
            tier=ModelTier.DEEPSEEK_V3,
            system=system_prompt,
            user=user_prompt,
            schema=QueryValidationBatch,
        )

        valid_queries = []
        for query, validation in zip(queries, result.validations):
            if validation.is_relevant:
                valid_queries.append(query)
            else:
                logger.warning(f"Query rejected: {query[:50]}... Reason: {validation.reason}")

        return valid_queries

    except Exception as e:
        logger.warning(f"Query validation failed: {e}, accepting all queries")
        return queries  # Fail open
