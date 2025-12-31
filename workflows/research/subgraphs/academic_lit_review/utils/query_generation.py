"""Query generation utilities for academic literature review workflow.

Contains:
- Academic search query generation prompts and functions
"""

import logging

from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
)

logger = logging.getLogger(__name__)


GENERATE_ACADEMIC_SEARCH_QUERIES_SYSTEM = """You are an expert academic researcher generating search queries for OpenAlex.

Generate 3-5 search queries optimized for finding relevant academic literature on the given topic.

Query strategies to use:
1. Core topic + methodology terms (e.g., "deep learning image classification survey")
2. Broader field + specific concepts (e.g., "computer vision neural networks review")
3. Related terminology / synonyms (e.g., "convolutional networks visual recognition")
4. Historical framing for seminal works (e.g., "artificial neural networks early foundations")
5. Recent framing for current research (e.g., "transformer architecture vision 2023")

Guidelines:
- Use academic/scholarly terminology
- Include methodological keywords when relevant
- Mix broad and specific queries
- Consider field-specific vocabulary
- Aim for queries that would find peer-reviewed literature

Output a JSON object:
{
  "queries": ["query1", "query2", "query3", ...]
}"""

GENERATE_ACADEMIC_SEARCH_QUERIES_USER = """Topic: {topic}

Research Questions:
{research_questions}

Focus Areas: {focus_areas}

Generate academic search queries to find relevant literature on this topic."""


async def generate_search_queries(
    topic: str,
    research_questions: list[str],
    focus_areas: list[str] | None = None,
    tier: ModelTier = ModelTier.HAIKU,
) -> list[str]:
    """Generate academic search queries for the given topic.

    Args:
        topic: Research topic
        research_questions: List of research questions
        focus_areas: Optional specific areas to focus on
        tier: Model tier for generation

    Returns:
        List of search queries
    """
    import json

    llm = get_llm(tier=tier)

    user_prompt = GENERATE_ACADEMIC_SEARCH_QUERIES_USER.format(
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
        focus_areas=", ".join(focus_areas) if focus_areas else "None specified",
    )

    try:
        response = await invoke_with_cache(
            llm,
            system_prompt=GENERATE_ACADEMIC_SEARCH_QUERIES_SYSTEM,
            user_prompt=user_prompt,
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)
        queries = result.get("queries", [])

        if not queries:
            # Fallback to topic as query
            queries = [topic]

        logger.info(f"Generated {len(queries)} search queries for topic: {topic[:50]}...")
        return queries

    except Exception as e:
        logger.error(f"Failed to generate search queries: {e}")
        # Fallback to basic queries
        return [topic, f"{topic} review", f"{topic} survey"]
