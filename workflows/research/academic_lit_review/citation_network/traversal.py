"""Graph traversal algorithms for citation network expansion."""

import asyncio
import logging
from typing import Any

from langchain_tools.openalex import get_forward_citations, get_backward_citations
from workflows.research.academic_lit_review.state import CitationEdge

from .types import CitationNetworkState, MAX_CITATIONS_PER_PAPER, MAX_CONCURRENT_FETCHES

logger = logging.getLogger(__name__)


async def fetch_forward_citations_node(state: CitationNetworkState) -> dict[str, Any]:
    """Fetch papers that cite the seed papers (forward citations).

    For each seed paper, retrieves papers that cite it, sorted by
    citation count to prioritize impactful citing works.
    """
    seed_dois = state.get("seed_dois", [])
    quality_settings = state["quality_settings"]
    min_citations = quality_settings.get("min_citations_filter", 10)

    if not seed_dois:
        logger.warning("No seed DOIs provided for forward citation search")
        return {"forward_results": [], "citation_edges": []}

    forward_results: list[dict] = []
    citation_edges: list[CitationEdge] = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def fetch_single_forward(
        seed_doi: str,
    ) -> tuple[list[dict], list[CitationEdge]]:
        """Fetch forward citations for a single seed DOI."""
        async with semaphore:
            try:
                result = await get_forward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                    min_citations=min_citations,
                )

                papers = []
                edges = []

                for work in result.results:
                    work_dict = (
                        work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    )
                    papers.append(work_dict)

                    if work_dict.get("doi"):
                        citing_doi = (
                            work_dict["doi"]
                            .replace("https://doi.org/", "")
                            .replace("http://doi.org/", "")
                        )
                        edges.append(
                            CitationEdge(
                                citing_doi=citing_doi,
                                cited_doi=seed_doi,
                                edge_type="forward",
                            )
                        )

                logger.debug(
                    f"Forward citations for {seed_doi[:30]}...: {len(papers)} results"
                )
                return papers, edges

            except Exception as e:
                logger.warning(f"Failed to fetch forward citations for {seed_doi}: {e}")
                return [], []

    tasks = [fetch_single_forward(doi) for doi in seed_dois]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Forward citation task failed: {result}")
            continue
        papers, edges = result
        forward_results.extend(papers)
        citation_edges.extend(edges)

    logger.info(
        f"Fetched {len(forward_results)} forward citations from {len(seed_dois)} seeds"
    )

    return {
        "forward_results": forward_results,
        "citation_edges": citation_edges,
    }


async def fetch_backward_citations_node(state: CitationNetworkState) -> dict[str, Any]:
    """Fetch papers cited by the seed papers (backward citations/references).

    For each seed paper, retrieves papers in its reference list.
    These often include seminal works in the field.
    """
    seed_dois = state.get("seed_dois", [])
    existing_edges = state.get("citation_edges", [])

    if not seed_dois:
        logger.warning("No seed DOIs provided for backward citation search")
        return {"backward_results": [], "citation_edges": existing_edges}

    backward_results: list[dict] = []
    new_edges: list[CitationEdge] = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def fetch_single_backward(
        seed_doi: str,
    ) -> tuple[list[dict], list[CitationEdge]]:
        """Fetch backward citations for a single seed DOI."""
        async with semaphore:
            try:
                result = await get_backward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                )

                papers = []
                edges = []

                for work in result.results:
                    work_dict = (
                        work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    )
                    papers.append(work_dict)

                    if work_dict.get("doi"):
                        cited_doi = (
                            work_dict["doi"]
                            .replace("https://doi.org/", "")
                            .replace("http://doi.org/", "")
                        )
                        edges.append(
                            CitationEdge(
                                citing_doi=seed_doi,
                                cited_doi=cited_doi,
                                edge_type="backward",
                            )
                        )

                logger.debug(
                    f"Backward citations for {seed_doi[:30]}...: {len(papers)} results"
                )
                return papers, edges

            except Exception as e:
                logger.warning(
                    f"Failed to fetch backward citations for {seed_doi}: {e}"
                )
                return [], []

    tasks = [fetch_single_backward(doi) for doi in seed_dois]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Backward citation task failed: {result}")
            continue
        papers, edges = result
        backward_results.extend(papers)
        new_edges.extend(edges)

    all_edges = existing_edges + new_edges

    logger.info(
        f"Fetched {len(backward_results)} backward citations from {len(seed_dois)} seeds"
    )

    return {
        "backward_results": backward_results,
        "citation_edges": all_edges,
    }
