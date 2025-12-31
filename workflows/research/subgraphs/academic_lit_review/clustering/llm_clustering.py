"""LLM-based semantic clustering implementation."""

import logging
from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import LLMTheme, LLMTopicSchema
from workflows.shared.llm_utils import ModelTier, get_llm

from .formatters import format_paper_for_llm
from .prompts import LLM_CLUSTERING_SYSTEM_PROMPT, LLM_CLUSTERING_USER_TEMPLATE
from .schemas import LLMTopicSchemaOutput

logger = logging.getLogger(__name__)


async def run_llm_clustering_node(state: dict) -> dict[str, Any]:
    """Semantic clustering using Claude Sonnet 4.5 with 1M context.

    Feeds ALL paper summaries to a single LLM call, leveraging the
    1M token context window to see the entire corpus at once.
    """
    paper_summaries = state.get("paper_summaries", {})
    input_data = state.get("input", {})

    if not paper_summaries:
        logger.warning("No paper summaries for LLM clustering")
        return {
            "llm_topic_schema": None,
            "llm_error": "No paper summaries available",
        }

    topic = input_data.get("topic", "Unknown topic")
    research_questions = input_data.get("research_questions", [])

    # Format all summaries for the prompt
    summaries_text = "\n\n---\n\n".join(
        format_paper_for_llm(doi, summary)
        for doi, summary in paper_summaries.items()
    )

    # Format research questions
    rq_text = "\n".join(f"- {q}" for q in research_questions) if research_questions else "None specified"

    user_prompt = LLM_CLUSTERING_USER_TEMPLATE.format(
        paper_count=len(paper_summaries),
        topic=topic,
        research_questions=rq_text,
        summaries=summaries_text,
    )

    try:
        # Use Sonnet with high max_tokens for 1M context handling
        llm = get_llm(
            tier=ModelTier.SONNET,
            max_tokens=16000,
        )

        # Use structured output to avoid JSON parsing issues
        structured_llm = llm.with_structured_output(LLMTopicSchemaOutput)
        messages = [
            {"role": "system", "content": LLM_CLUSTERING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result: LLMTopicSchemaOutput = await structured_llm.ainvoke(messages)

        # Convert Pydantic model to TypedDict for state compatibility
        themes: list[LLMTheme] = []
        for theme in result.themes:
            themes.append(
                LLMTheme(
                    name=theme.name,
                    description=theme.description,
                    paper_dois=theme.paper_dois,
                    sub_themes=theme.sub_themes,
                    relationships=theme.relationships,
                )
            )

        llm_schema = LLMTopicSchema(
            themes=themes,
            reasoning=result.reasoning,
        )

        # Log detailed theme information for diagnostics
        logger.info(
            f"LLM clustering complete: {len(themes)} themes "
            f"from {len(paper_summaries)} papers"
        )
        for t in themes[:5]:  # Log first 5 themes for diagnostics
            logger.info(
                f"  Theme '{t['name']}': {len(t['paper_dois'])} papers"
            )

        return {
            "llm_topic_schema": llm_schema,
            "llm_error": None,
        }

    except Exception as e:
        error_msg = f"LLM clustering failed: {str(e)}"
        logger.error(error_msg)
        return {
            "llm_topic_schema": None,
            "llm_error": error_msg,
        }
