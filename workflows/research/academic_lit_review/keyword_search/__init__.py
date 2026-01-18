"""Keyword search subgraph for academic literature discovery.

Generates academic search queries, executes them against OpenAlex,
and filters results by relevance scoring.

Flow:
    START -> generate_queries -> search_openalex -> filter_by_relevance -> END
"""

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    QualitySettings,
)
from workflows.shared.language import LanguageConfig

from .types import KeywordSearchState
from .query_builder import generate_queries_node
from .searcher import search_openalex_node, filter_by_relevance_node


def create_keyword_search_subgraph() -> StateGraph:
    """Create the keyword search subgraph.

    Flow:
        START -> generate_queries -> search_openalex -> filter_by_relevance -> END
    """
    builder = StateGraph(KeywordSearchState)

    builder.add_node("generate_queries", generate_queries_node)
    builder.add_node("search_openalex", search_openalex_node)
    builder.add_node("filter_by_relevance", filter_by_relevance_node)

    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "search_openalex")
    builder.add_edge("search_openalex", "filter_by_relevance")
    builder.add_edge("filter_by_relevance", END)

    return builder.compile()


keyword_search_subgraph = create_keyword_search_subgraph()


async def run_keyword_search(
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    date_range: Optional[tuple[int, int]] = None,
    language_config: Optional[LanguageConfig] = None,
) -> dict[str, Any]:
    """Run keyword search discovery as a standalone operation.

    Args:
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        date_range: Optional (start_year, end_year) filter
        language_config: Optional language configuration

    Returns:
        Dict with discovered_papers, rejected_papers, keyword_dois
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=date_range,
        language_code=language_config["code"] if language_config else "en",
    )

    initial_state = KeywordSearchState(
        input=input_data,
        quality_settings=quality_settings,
        language_config=language_config,
        search_queries=[],
        raw_results=[],
        discovered_papers=[],
        rejected_papers=[],
        keyword_dois=[],
    )

    result = await keyword_search_subgraph.ainvoke(initial_state)
    return {
        "discovered_papers": result.get("discovered_papers", []),
        "rejected_papers": result.get("rejected_papers", []),
        "keyword_dois": result.get("keyword_dois", []),
    }


__all__ = [
    "KeywordSearchState",
    "keyword_search_subgraph",
    "run_keyword_search",
]
