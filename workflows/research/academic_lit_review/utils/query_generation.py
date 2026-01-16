"""Query generation utilities for academic literature review workflow.

Contains:
- Academic search query generation prompts and functions
"""

import logging

from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier, get_structured_output
from workflows.shared.language import LanguageConfig, get_translated_prompt, translate_queries

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

Generate academic search queries to find relevant literature on this topic."""


class AcademicQueryResponse(BaseModel):
    """Structured output for academic search queries."""

    queries: list[str] = Field(
        description="3-5 search queries optimized for academic literature search",
        min_length=1,
    )


async def generate_search_queries(
    topic: str,
    research_questions: list[str],
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.HAIKU,
) -> list[str]:
    """Generate academic search queries for the given topic.

    Args:
        topic: Research topic
        research_questions: List of research questions
        language_config: Optional language configuration for translation
        tier: Model tier for generation

    Returns:
        List of search queries
    """
    # Translate system prompt if needed
    system_prompt = GENERATE_ACADEMIC_SEARCH_QUERIES_SYSTEM
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            GENERATE_ACADEMIC_SEARCH_QUERIES_SYSTEM,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="lit_review_query_gen_system",
        )

    user_prompt = GENERATE_ACADEMIC_SEARCH_QUERIES_USER.format(
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
    )

    try:
        result: AcademicQueryResponse = await get_structured_output(
            output_schema=AcademicQueryResponse,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tier=tier,
        )

        queries = result.queries

        if not queries:
            queries = [topic]

        # Translate queries to target language for non-English searches.
        # Testing shows that Spanish queries find papers with Spanish content,
        # while English queries + language metadata filter find papers with
        # missing abstracts or English content in Spanish-language journals.
        if language_config and language_config["code"] != "en":
            queries = await translate_queries(
                queries,
                target_language_code=language_config["code"],
                target_language_name=language_config["name"],
            )
            logger.info(
                f"Translated {len(queries)} queries to {language_config['name']}"
            )

        logger.info(
            f"Generated {len(queries)} search queries for topic: {topic[:50]}..."
        )
        return queries

    except Exception as e:
        logger.error(f"Failed to generate search queries: {e}")
        # Fallback to basic queries
        return [topic, f"{topic} review", f"{topic} survey"]
