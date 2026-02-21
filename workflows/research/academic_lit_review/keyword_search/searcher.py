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

    Uses a three-phase search with fallback to ensure emerging work isn't filtered out:
    - Phase 1: Primary recent (past recency_years): No citation threshold
    - Phase 1b: Fallback recent (recency_years to recency_years_fallback): No citation
      threshold, only triggered if Phase 1 yields too few results for the recency quota
    - Phase 2: Older papers: Normal citation threshold from quality settings

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
    recency_years = quality_settings.get("recency_years", 1)
    recency_years_fallback = quality_settings.get("recency_years_fallback", 2)
    recency_quota = quality_settings.get("recency_quota", 0.35)
    max_papers = quality_settings.get("max_papers", 100)
    date_range = input_data.get("date_range")
    language_code = language_config["code"] if language_config else None

    # Calculate cutoffs
    current_year = datetime.now(timezone.utc).year
    recent_cutoff = current_year - recency_years
    fallback_cutoff = current_year - recency_years_fallback

    # Handle user-specified date range constraints
    user_from_year = date_range[0] if date_range else None
    user_to_year = date_range[1] if date_range else None

    all_results: list[OpenAlexWork] = []

    async def _search(query: str, from_year: int | None, to_year: int | None, min_cites: int) -> list[OpenAlexWork]:
        """Search OpenAlex for a single query within a year range."""
        try:
            result = await openalex_search.ainvoke(
                {
                    "query": query,
                    "limit": MAX_RESULTS_PER_QUERY,
                    "min_citations": min_cites,
                    "from_year": from_year,
                    "to_year": to_year,
                    "language": language_code,
                }
            )
            works = result.get("results", [])
            logger.debug(f"Query '{query[:40]}...' [{from_year}-{to_year}] min_cites={min_cites}: {len(works)} results")
            return works
        except Exception as e:
            logger.warning(f"OpenAlex search failed for '{query}': {e}")
            return []

    async def _gather(tasks: list) -> list[OpenAlexWork]:
        results: list[OpenAlexWork] = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")
            else:
                results.extend(result)
        return results

    # Phase 1: Primary recent window (no citation threshold)
    recent_from = max(recent_cutoff, user_from_year) if user_from_year else recent_cutoff
    recent_to = user_to_year

    if recent_to is None or recent_to >= recent_cutoff:
        phase1 = await _gather([_search(q, recent_from, recent_to, 0) for q in queries])
        all_results.extend(phase1)
        logger.info(f"Phase 1 (primary recent >= {recent_cutoff}): {len(phase1)} results")

    # Phase 1b: Fallback band if primary recent yielded too few for the quota
    effective_recent_cutoff = recent_cutoff
    target_recent = int(max_papers * recency_quota)

    if len(all_results) < target_recent and fallback_cutoff < recent_cutoff:
        fb_from = max(fallback_cutoff, user_from_year) if user_from_year else fallback_cutoff
        fb_to = min(recent_cutoff - 1, user_to_year) if user_to_year else recent_cutoff - 1

        if fb_from <= fb_to:
            logger.info(
                f"Recency fallback: {len(all_results)} primary recent < target {target_recent}, "
                f"widening to {fb_from}-{fb_to}"
            )
            phase1b = await _gather([_search(q, fb_from, fb_to, 0) for q in queries])
            all_results.extend(phase1b)
            effective_recent_cutoff = fallback_cutoff
            logger.info(f"Phase 1b (fallback {fb_from}-{fb_to}): {len(phase1b)} results")

    # Phase 2: Older papers with citation threshold
    older_to = min(effective_recent_cutoff - 1, user_to_year) if user_to_year else effective_recent_cutoff - 1
    older_from = user_from_year

    if older_from is None or older_from < effective_recent_cutoff:
        phase2 = await _gather([_search(q, older_from, older_to, min_citations) for q in queries])
        all_results.extend(phase2)
        logger.info(f"Phase 2 (older < {effective_recent_cutoff}, min_cites={min_citations}): {len(phase2)} results")

    logger.info(
        f"OpenAlex keyword search: {len(all_results)} raw results from {len(queries)} queries "
        f"(recent>={effective_recent_cutoff} no citation filter, older<{effective_recent_cutoff} min_citations={min_citations})"
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
        "discovered_papers": selected + overflow + fallback_candidates,
        "rejected_papers": rejected,
        "fallback_queue": fallback_queue,
        "keyword_dois": keyword_dois,
    }
