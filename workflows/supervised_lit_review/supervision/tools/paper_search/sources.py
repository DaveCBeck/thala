"""Different search sources for paper retrieval."""

import logging

from langchain_tools.base import get_store_manager
from workflows.academic_lit_review.state import PaperSummary

logger = logging.getLogger(__name__)


async def semantic_search(
    query: str,
    paper_summaries: dict[str, PaperSummary],
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Semantic search using embeddings.

    Returns list of (doi, similarity_score) tuples.
    """
    store_manager = get_store_manager()

    # Get DOIs that have ES records (can be searched in Chroma)
    searchable_dois = [
        doi for doi, s in paper_summaries.items() if s.get("es_record_id")
    ]

    if not searchable_dois:
        return []

    try:
        query_embedding = await store_manager.embedding.embed(query)

        # Search Chroma for semantically similar content
        results = await store_manager.chroma.search(
            query_embedding=query_embedding,
            n_results=limit * 2,  # Get extra for filtering
            where=None,  # No filter - search all
        )

        # Map results back to DOIs via zotero_key
        doi_scores: list[tuple[str, float]] = []
        key_to_doi = {
            s.get("zotero_key"): doi
            for doi, s in paper_summaries.items()
            if s.get("zotero_key")
        }

        for r in results:
            metadata = r.get("metadata", {})
            zotero_key = metadata.get("zotero_key")
            if zotero_key and zotero_key in key_to_doi:
                doi = key_to_doi[zotero_key]
                similarity = 1 - r.get("distance", 1)  # Convert distance to similarity
                doi_scores.append((doi, similarity))

        return doi_scores[:limit]

    except Exception as e:
        logger.warning(f"Semantic search failed: {e}")
        return []


async def keyword_search(
    query: str,
    paper_summaries: dict[str, PaperSummary],
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Keyword search using Elasticsearch text matching.

    Returns list of (doi, relevance_score) tuples.
    """
    store_manager = get_store_manager()

    try:
        es_query = {"match": {"content": query}}
        records = await store_manager.es_stores.store.search(
            query=es_query,
            size=limit * 2,
        )

        # Map results back to DOIs
        zotero_key_to_doi = {
            s.get("zotero_key"): doi
            for doi, s in paper_summaries.items()
            if s.get("zotero_key")
        }

        doi_scores: list[tuple[str, float]] = []
        for r in records:
            if r.zotero_key and r.zotero_key in zotero_key_to_doi:
                doi = zotero_key_to_doi[r.zotero_key]
                # Normalize score to 0-1 range (ES scores vary widely)
                score = min(1.0, getattr(r, "_score", 1.0) / 10.0)
                doi_scores.append((doi, score))

        return doi_scores[:limit]

    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []


def merge_search_results(
    semantic_results: list[tuple[str, float]],
    keyword_results: list[tuple[str, float]],
    limit: int = 10,
) -> list[tuple[str, float]]:
    """Merge and rank results from both search methods.

    Uses reciprocal rank fusion for fair combination.
    """
    scores: dict[str, float] = {}
    k = 60  # RRF constant

    # Add semantic scores using RRF
    for rank, (doi, _) in enumerate(semantic_results):
        scores[doi] = scores.get(doi, 0) + 1.0 / (k + rank + 1)

    # Add keyword scores using RRF
    for rank, (doi, _) in enumerate(keyword_results):
        scores[doi] = scores.get(doi, 0) + 1.0 / (k + rank + 1)

    # Sort by combined score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:limit]
