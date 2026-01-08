"""Query generation utilities."""

import logging
from typing import Any

from workflows.research.state import ResearcherState, SearchQueries
from workflows.research.prompts import (
    GENERATE_WEB_QUERIES_SYSTEM,
    GENERATE_ACADEMIC_QUERIES_SYSTEM,
    GENERATE_BOOK_QUERIES_SYSTEM,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

from .query_validator import validate_queries

logger = logging.getLogger(__name__)

# Map researcher types to their specialized prompts
RESEARCHER_QUERY_PROMPTS = {
    "web": GENERATE_WEB_QUERIES_SYSTEM,
    "academic": GENERATE_ACADEMIC_QUERIES_SYSTEM,
    "book": GENERATE_BOOK_QUERIES_SYSTEM,
}


def create_generate_queries(researcher_type: str = "web"):
    """Create a generate_queries node function for a specific researcher type.

    Args:
        researcher_type: One of "web", "academic", "book"

    Returns:
        Async function that generates queries optimized for the researcher type.
    """

    async def generate_queries(state: ResearcherState) -> dict[str, Any]:
        """Generate search queries using structured output with validation.

        Uses researcher-type-specific prompts to optimize queries for:
        - Web: General search engines (Firecrawl, Perplexity)
        - Academic: OpenAlex peer-reviewed literature
        - Book: book_search databases

        If language_config is set, generates queries in the target language
        for better search results in that language.
        """
        question = state["question"]
        language_config = state.get("language_config")

        # Get researcher-specific base prompt
        base_prompt = RESEARCHER_QUERY_PROMPTS.get(researcher_type, GENERATE_WEB_QUERIES_SYSTEM)

        # Build language-aware prompt
        if language_config and language_config["code"] != "en":
            lang_name = language_config["name"]
            prompt = f"""{base_prompt}

Generate queries in {lang_name} to find {lang_name}-language sources.
Write queries naturally in {lang_name}.

Question: {question['question']}
"""
        else:
            prompt = f"""{base_prompt}

Question: {question['question']}
"""

        try:
            result: SearchQueries = await get_structured_output(
                output_schema=SearchQueries,
                user_prompt=prompt,
                tier=ModelTier.HAIKU,
            )

            # Validate queries are relevant
            valid_queries = await validate_queries(
                queries=result.queries,
                research_question=question['question'],
                research_brief=question.get('brief'),
                draft_notes=question.get('context'),
            )

            if not valid_queries:
                logger.warning("All queries invalid, using fallback")
                valid_queries = [question["question"]]

            lang_info = f" ({language_config['code']})" if language_config else ""
            logger.debug(
                f"Generated {len(valid_queries)} {researcher_type} queries{lang_info} "
                f"for: {question['question'][:50]}..."
            )
            return {"search_queries": valid_queries}

        except Exception as e:
            logger.error(f"Failed to generate {researcher_type} queries: {e}")
            # Fallback: use the question as the query
            return {"search_queries": [question["question"]]}

    return generate_queries


# Backwards compatibility: default web query generator
async def generate_queries(state: ResearcherState) -> dict[str, Any]:
    """Generate search queries (default: web-optimized).

    For specialized query generation, use create_generate_queries(researcher_type).
    """
    generator = create_generate_queries("web")
    return await generator(state)
