"""Query generation for keyword search."""

import logging
from typing import Any

from workflows.academic_lit_review.utils import generate_search_queries
from workflows.shared.llm_utils import ModelTier

from .types import KeywordSearchState, MAX_QUERIES

logger = logging.getLogger(__name__)


async def generate_queries_node(state: KeywordSearchState) -> dict[str, Any]:
    """Generate academic search queries for the topic.

    Uses LLM to generate multiple query variations:
    1. Core topic + methodology terms
    2. Broader field + specific concepts
    3. Related terminology / synonyms
    4. Historical + recent framing
    """
    input_data = state["input"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    focus_areas = input_data.get("focus_areas")
    language_config = state.get("language_config")

    queries = await generate_search_queries(
        topic=topic,
        research_questions=research_questions,
        focus_areas=focus_areas,
        language_config=language_config,
        tier=ModelTier.HAIKU,
    )

    queries = queries[:MAX_QUERIES]

    logger.info(f"Generated {len(queries)} keyword search queries")
    return {"search_queries": queries}
