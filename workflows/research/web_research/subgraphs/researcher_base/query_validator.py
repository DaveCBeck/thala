"""Query validation utilities."""

import logging

from workflows.research.web_research.state import QueryValidationBatch
from workflows.shared.llm_utils import ModelTier, get_structured_output

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
            context_parts.append(
                f"Objectives: {', '.join(research_brief['objectives'][:3])}"
            )
    if draft_notes:
        context_parts.append(f"Current Notes: {draft_notes[:500]}...")

    context = "\n".join(context_parts)
    queries_list = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))

    prompt = f"""Validate whether these search queries are relevant to the research task.

{context}

Proposed Search Queries:
{queries_list}

For each query, determine if it's actually relevant to the research question above.
Reject queries that:
- Contain system metadata (iteration counts, percentages, internal state)
- Are about completely unrelated topics
- Are too vague or generic to be useful

Accept queries that would help find information about the research topic.
"""

    try:
        result: QueryValidationBatch = await get_structured_output(
            output_schema=QueryValidationBatch,
            user_prompt=prompt,
            tier=ModelTier.HAIKU,
        )

        valid_queries = []
        for query, validation in zip(queries, result.validations):
            if validation.is_relevant:
                valid_queries.append(query)
            else:
                logger.warning(
                    f"Query rejected: {query[:50]}... Reason: {validation.reason}"
                )

        return valid_queries

    except Exception as e:
        logger.warning(f"Query validation failed: {e}, accepting all queries")
        return queries  # Fail open
