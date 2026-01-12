"""Search execution and relevance filtering for keyword search."""

import asyncio
import logging
from typing import Any

from langchain_tools.openalex import openalex_search, OpenAlexWork
from workflows.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    batch_score_relevance,
)
from workflows.shared.llm_utils import ModelTier

from .types import KeywordSearchState, MAX_RESULTS_PER_QUERY

logger = logging.getLogger(__name__)


async def search_openalex_node(state: KeywordSearchState) -> dict[str, Any]:
    """Execute searches against OpenAlex with quality filters.

    Filters applied:
    - Minimum citation count (from quality settings)
    - Date range (if specified in input)
    - Language filter (if language_config specified)
    - Sorts by relevance score
    """
    queries = state.get("search_queries", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    language_config = state.get("language_config")

    min_citations = quality_settings.get("min_citations_filter", 10)
    date_range = input_data.get("date_range")
    from_year = date_range[0] if date_range else None
    to_year = date_range[1] if date_range else None
    language_code = language_config["code"] if language_config else None

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
                "language": language_code,
            })

            works = []
            for r in result.get("results", []):
                works.append(r)

            logger.debug(f"Query '{query[:40]}...' returned {len(works)} results")
            return works

        except Exception as e:
            logger.warning(f"OpenAlex search failed for '{query}': {e}")
            return []

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

    papers = []
    for result in raw_results:
        paper = convert_to_paper_metadata(
            work=result,
            discovery_stage=0,
            discovery_method="keyword",
        )
        if paper:
            papers.append(paper)

    papers = deduplicate_papers(papers)

    if not papers:
        logger.warning("No papers after conversion and deduplication")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "keyword_dois": [],
        }

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
