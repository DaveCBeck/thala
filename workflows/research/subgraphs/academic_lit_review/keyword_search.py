"""Keyword search subgraph for academic literature discovery.

Generates academic search queries, executes them against OpenAlex,
and filters results by relevance scoring.

Flow:
    START -> generate_queries -> search_openalex -> filter_by_relevance -> END
"""

import asyncio
import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langchain_tools.openalex import openalex_search, OpenAlexWork
from workflows.research.subgraphs.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.research.subgraphs.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    generate_search_queries,
    batch_score_relevance,
)
from workflows.shared.llm_utils import ModelTier
from workflows.shared.language import LanguageConfig

logger = logging.getLogger(__name__)

# Constants
MAX_QUERIES = 5
MAX_RESULTS_PER_QUERY = 25


# =============================================================================
# State Definition
# =============================================================================


class KeywordSearchState(TypedDict):
    """State for keyword-based academic search subgraph."""

    # Input (from parent)
    input: LitReviewInput
    quality_settings: QualitySettings
    language_config: Optional[LanguageConfig]

    # Internal state
    search_queries: list[str]
    raw_results: list[OpenAlexWork]  # Raw OpenAlex results

    # Output
    discovered_papers: list[PaperMetadata]  # After relevance filtering
    rejected_papers: list[PaperMetadata]  # Papers that didn't pass filter
    keyword_dois: list[str]  # DOIs of discovered papers


# =============================================================================
# Node Functions
# =============================================================================


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

    # Limit queries
    queries = queries[:MAX_QUERIES]

    logger.info(f"Generated {len(queries)} keyword search queries")
    return {"search_queries": queries}


async def search_openalex_node(state: KeywordSearchState) -> dict[str, Any]:
    """Execute searches against OpenAlex with quality filters.

    Filters applied:
    - Minimum citation count (from quality settings)
    - Date range (if specified in input)
    - Sorts by relevance score
    """
    queries = state.get("search_queries", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]

    # Get filter parameters
    min_citations = quality_settings.get("min_citations_filter", 10)
    date_range = input_data.get("date_range")
    from_year = date_range[0] if date_range else None
    to_year = date_range[1] if date_range else None

    all_results: list[OpenAlexWork] = []

    async def search_single_query(query: str) -> list[OpenAlexWork]:
        """Search OpenAlex for a single query."""
        try:
            result = await openalex_search.ainvoke({
                "query": query,
                "limit": MAX_RESULTS_PER_QUERY,
                "min_citations": min_citations,
                "from_year": from_year,
                "to_year": to_year,
            })

            # Convert results to OpenAlexWork objects
            works = []
            for r in result.get("results", []):
                works.append(r)  # Already dict from openalex_search

            logger.debug(f"Query '{query[:40]}...' returned {len(works)} results")
            return works

        except Exception as e:
            logger.warning(f"OpenAlex search failed for '{query}': {e}")
            return []

    # Execute all searches in parallel
    search_tasks = [search_single_query(q) for q in queries]
    results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

    for result in results_lists:
        if isinstance(result, Exception):
            logger.error(f"Search task failed: {result}")
            continue
        all_results.extend(result)

    logger.info(
        f"OpenAlex keyword search: {len(all_results)} raw results from {len(queries)} queries"
    )

    return {"raw_results": all_results}


async def filter_by_relevance_node(state: KeywordSearchState) -> dict[str, Any]:
    """Filter results by LLM-based relevance scoring.

    Converts raw OpenAlex results to PaperMetadata, deduplicates,
    and scores relevance to the research topic.
    """
    raw_results = state.get("raw_results", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    language_config = state.get("language_config")

    if not raw_results:
        logger.warning("No raw results to filter")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "keyword_dois": [],
        }

    # Convert to PaperMetadata
    papers = []
    for result in raw_results:
        paper = convert_to_paper_metadata(
            work=result,
            discovery_stage=0,
            discovery_method="keyword",
        )
        if paper:
            papers.append(paper)

    # Deduplicate
    papers = deduplicate_papers(papers)

    if not papers:
        logger.warning("No papers after conversion and deduplication")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "keyword_dois": [],
        }

    # Score relevance
    relevant, rejected = await batch_score_relevance(
        papers=papers,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        language_config=language_config,
        tier=ModelTier.HAIKU,
        max_concurrent=10,
        use_batch_api=quality_settings.get("use_batch_api", True),
    )

    # Extract DOIs
    keyword_dois = [p.get("doi") for p in relevant if p.get("doi")]

    logger.info(
        f"Keyword search discovered {len(relevant)} relevant papers "
        f"(rejected {len(rejected)})"
    )

    return {
        "discovered_papers": relevant,
        "rejected_papers": rejected,
        "keyword_dois": keyword_dois,
    }


# =============================================================================
# Subgraph Definition
# =============================================================================


def create_keyword_search_subgraph() -> StateGraph:
    """Create the keyword search subgraph.

    Flow:
        START -> generate_queries -> search_openalex -> filter_by_relevance -> END
    """
    builder = StateGraph(KeywordSearchState)

    # Add nodes
    builder.add_node("generate_queries", generate_queries_node)
    builder.add_node("search_openalex", search_openalex_node)
    builder.add_node("filter_by_relevance", filter_by_relevance_node)

    # Add edges
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "search_openalex")
    builder.add_edge("search_openalex", "filter_by_relevance")
    builder.add_edge("filter_by_relevance", END)

    return builder.compile()


# Export compiled subgraph
keyword_search_subgraph = create_keyword_search_subgraph()


# =============================================================================
# Convenience Function
# =============================================================================


async def run_keyword_search(
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    date_range: Optional[tuple[int, int]] = None,
    focus_areas: Optional[list[str]] = None,
    language_config: Optional[LanguageConfig] = None,
) -> dict[str, Any]:
    """Run keyword search discovery as a standalone operation.

    Args:
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        date_range: Optional (start_year, end_year) filter
        focus_areas: Optional specific areas to focus on
        language_config: Optional language configuration

    Returns:
        Dict with discovered_papers, rejected_papers, keyword_dois
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=date_range,
        include_books=False,
        focus_areas=focus_areas,
        exclude_terms=None,
        max_papers=None,
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
