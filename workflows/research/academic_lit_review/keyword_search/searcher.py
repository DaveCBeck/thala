"""Search execution and relevance filtering for keyword search."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_tools.openalex import openalex_search, OpenAlexWork
from workflows.research.academic_lit_review.state import FallbackCandidate
from workflows.research.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    batch_score_relevance,
)
from workflows.shared.llm_utils import ModelTier
from workflows.shared.language import filter_by_content_language

from .types import KeywordSearchState, MAX_RESULTS_PER_QUERY

logger = logging.getLogger(__name__)


async def search_openalex_node(state: KeywordSearchState) -> dict[str, Any]:
    """Execute searches against OpenAlex with recency-aware citation thresholds.

    Uses two-phase search to ensure emerging work isn't filtered out:
    - Recent papers (past N years): No citation threshold
    - Older papers: Normal citation threshold from quality settings

    Other filters applied:
    - Date range (if specified in input)
    - Language filter (if language_config specified)
    - Sorts by relevance score
    """
    queries = state.get("search_queries", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    language_config = state.get("language_config")

    min_citations = quality_settings.get("min_citations_filter", 10)
    recency_years = quality_settings.get("recency_years", 3)
    date_range = input_data.get("date_range")
    language_code = language_config["code"] if language_config else None

    # Calculate recency cutoff
    current_year = datetime.now(timezone.utc).year
    recent_cutoff = current_year - recency_years

    # Handle user-specified date range constraints
    user_from_year = date_range[0] if date_range else None
    user_to_year = date_range[1] if date_range else None

    all_results: list[OpenAlexWork] = []

    async def search_single_query(query: str) -> list[OpenAlexWork]:
        """Search OpenAlex for a single query with two-phase approach."""
        works = []

        try:
            # Phase 1: Recent papers with relaxed citation threshold
            recent_from = max(recent_cutoff, user_from_year) if user_from_year else recent_cutoff
            recent_to = user_to_year  # Respect user's upper bound

            # Only search recent if the date range allows it
            if recent_to is None or recent_to >= recent_cutoff:
                recent_result = await openalex_search.ainvoke(
                    {
                        "query": query,
                        "limit": MAX_RESULTS_PER_QUERY,
                        "min_citations": 0,  # No citation requirement for recent
                        "from_year": recent_from,
                        "to_year": recent_to,
                        "language": language_code,
                    }
                )
                for r in recent_result.get("results", []):
                    works.append(r)
                logger.debug(
                    f"Query '{query[:40]}...' recent phase: {len(recent_result.get('results', []))} results"
                )

            # Phase 2: Older papers with normal citation threshold
            older_to = min(recent_cutoff - 1, user_to_year) if user_to_year else recent_cutoff - 1
            older_from = user_from_year  # Respect user's lower bound

            # Only search older if the date range allows it
            if older_from is None or older_from < recent_cutoff:
                older_result = await openalex_search.ainvoke(
                    {
                        "query": query,
                        "limit": MAX_RESULTS_PER_QUERY,
                        "min_citations": min_citations,
                        "from_year": older_from,
                        "to_year": older_to,
                        "language": language_code,
                    }
                )
                for r in older_result.get("results", []):
                    works.append(r)
                logger.debug(
                    f"Query '{query[:40]}...' older phase: {len(older_result.get('results', []))} results"
                )

            logger.debug(f"Query '{query[:40]}...' total: {len(works)} results")
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
        f"OpenAlex keyword search: {len(all_results)} raw results from {len(queries)} queries "
        f"(two-phase: recent>={recent_cutoff} no citation filter, older<{recent_cutoff} min_citations={min_citations})"
    )

    return {"raw_results": all_results}


async def filter_by_relevance_node(state: KeywordSearchState) -> dict[str, Any]:
    """Filter results by LLM-based relevance scoring.

    Converts raw OpenAlex results to PaperMetadata, deduplicates,
    and scores relevance to the research topic. Creates a fallback queue
    from overflow papers (above threshold but excluded by max_papers) and
    near-threshold papers (0.5-0.6 score).
    """
    raw_results = state.get("raw_results", [])
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    language_config = state.get("language_config")
    max_papers = quality_settings.get("max_papers", 100)

    if not raw_results:
        logger.warning("No raw results to filter")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "fallback_queue": [],
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
            "fallback_queue": [],
            "keyword_dois": [],
        }

    # Content-based language filtering (secondary check after query translation)
    # Catches papers where metadata says target language but content is different
    if language_config and language_config.get("code") != "en":
        papers, lang_rejected = filter_by_content_language(
            papers,
            target_language=language_config["code"],
            text_fields=["abstract", "title"],
        )
        if lang_rejected:
            logger.info(
                f"Content language filter: kept {len(papers)}, "
                f"rejected {len(lang_rejected)} (abstract not in {language_config['code']})"
            )

        if not papers:
            logger.warning("No papers after language filtering")
            return {
                "discovered_papers": [],
                "rejected_papers": [],
                "fallback_queue": [],
                "keyword_dois": [],
            }

    relevant, fallback_candidates, rejected = await batch_score_relevance(
        papers=papers,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        fallback_threshold=0.5,
        language_config=language_config,
        tier=ModelTier.DEEPSEEK_V3,
        max_concurrent=10,
        use_batch_api=quality_settings.get("use_batch_api", True),
    )

    # Sort relevant papers by relevance score descending
    relevant.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)

    # Apply max_papers limit - overflow papers become highest-priority fallbacks
    if len(relevant) > max_papers:
        selected = relevant[:max_papers]
        overflow = relevant[max_papers:]
        logger.info(
            f"max_papers limit applied: keeping {len(selected)}, "
            f"{len(overflow)} overflow papers added to fallback queue"
        )
    else:
        selected = relevant
        overflow = []

    # Build fallback queue: overflow papers first (highest relevance), then near-threshold
    # Overflow papers scored >= 0.6, fallback_candidates scored 0.5-0.6
    fallback_queue: list[FallbackCandidate] = []

    for paper in overflow:
        fallback_queue.append(
            FallbackCandidate(
                doi=paper.get("doi", ""),
                relevance_score=paper.get("relevance_score", 0.6),
                source="overflow",
            )
        )

    for paper in fallback_candidates:
        fallback_queue.append(
            FallbackCandidate(
                doi=paper.get("doi", ""),
                relevance_score=paper.get("relevance_score", 0.5),
                source="near_threshold",
            )
        )

    keyword_dois = [p.get("doi") for p in selected if p.get("doi")]

    logger.info(
        f"Keyword search discovered {len(selected)} relevant papers "
        f"(fallback queue: {len(fallback_queue)}, rejected: {len(rejected)})"
    )

    return {
        "discovered_papers": selected,
        "rejected_papers": rejected,
        "fallback_queue": fallback_queue,
        "keyword_dois": keyword_dois,
    }
