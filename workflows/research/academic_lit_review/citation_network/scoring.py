"""Citation scoring and filtering functions."""

import logging
from typing import Any

from workflows.research.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    batch_score_relevance,
)
from workflows.shared.llm_utils import ModelTier
from workflows.shared.language import filter_by_content_language

from .types import CitationNetworkState

logger = logging.getLogger(__name__)


async def merge_and_filter_node(state: CitationNetworkState) -> dict[str, Any]:
    """Merge forward/backward results, deduplicate, and filter by relevance.

    Combines all discovered papers, removes duplicates and papers already
    in corpus, then scores relevance to filter down to useful additions.
    """
    forward_results = state.get("forward_results", [])
    backward_results = state.get("backward_results", [])
    existing_dois = state.get("existing_dois", set())
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    citation_edges = state.get("citation_edges", [])

    all_results = forward_results + backward_results

    if not all_results:
        logger.warning("No citation results to merge and filter")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "discovered_dois": [],
            "new_edges": citation_edges,
        }

    # Filter by content language early (before expensive relevance scoring)
    # Uses abstract text detection instead of unreliable metadata
    language_config = state.get("language_config")
    if language_config and language_config.get("code") != "en":
        target_lang = language_config["code"]
        pre_filter_count = len(all_results)
        all_results, lang_rejected = filter_by_content_language(
            all_results,
            target_language=target_lang,
            text_fields=["abstract", "title"],
        )
        if lang_rejected:
            logger.info(
                f"Content language filter ({target_lang}): kept {len(all_results)}/{pre_filter_count} "
                f"citations (rejected {len(lang_rejected)} with non-{target_lang} abstracts)"
            )

        if not all_results:
            logger.warning(f"No citations in target language ({target_lang})")
            return {
                "discovered_papers": [],
                "rejected_papers": [],
                "discovered_dois": [],
                "new_edges": citation_edges,
            }

    papers = []
    for result in all_results:
        discovery_method = "citation"

        paper = convert_to_paper_metadata(
            work=result,
            discovery_stage=1,
            discovery_method=discovery_method,
        )
        if paper:
            papers.append(paper)

    papers = deduplicate_papers(papers, existing_dois)

    if not papers:
        logger.warning("No new papers after deduplication")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "discovered_dois": [],
            "new_edges": citation_edges,
        }

    logger.info(
        f"Merged {len(all_results)} raw results to {len(papers)} unique new papers"
    )

    # language_config already fetched above for early filtering
    if language_config is None:
        language_config = state.get("language_config")
    # Note: fallback_candidates from citation network are not currently used,
    # but we accept the 3-tuple return for API consistency
    relevant, _fallback_candidates, rejected = await batch_score_relevance(
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

    discovered_dois = [p.get("doi") for p in relevant if p.get("doi")]

    discovered_doi_set = set(discovered_dois)
    seed_dois_set = set(state.get("seed_dois", []))
    valid_dois = discovered_doi_set | seed_dois_set

    filtered_edges = [
        edge
        for edge in citation_edges
        if edge.get("citing_doi") in valid_dois or edge.get("cited_doi") in valid_dois
    ]

    logger.info(
        f"Citation network discovered {len(relevant)} relevant papers "
        f"(rejected {len(rejected)}), {len(filtered_edges)} edges"
    )

    return {
        "discovered_papers": relevant,
        "rejected_papers": rejected,
        "discovered_dois": discovered_dois,
        "new_edges": filtered_edges,
    }
