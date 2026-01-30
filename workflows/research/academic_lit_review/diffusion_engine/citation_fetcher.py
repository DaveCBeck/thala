"""Citation fetching from OpenAlex API."""

import asyncio
import logging
from datetime import datetime, timezone

from langchain_tools.openalex import get_forward_citations, get_backward_citations
from workflows.research.academic_lit_review.state import CitationEdge
from .types import MAX_CITATIONS_PER_PAPER, MAX_CONCURRENT_FETCHES

logger = logging.getLogger(__name__)


async def fetch_citations_raw(
    seed_dois: list[str],
    min_citations: int = 10,
    recency_years: int = 3,
) -> tuple[list[dict], list[CitationEdge]]:
    """Fetch forward and backward citations with recency-aware thresholds.

    Uses two-phase forward citation fetching to ensure emerging work
    isn't filtered out:
    - Recent papers (past N years): No citation threshold
    - Older papers: Normal citation threshold

    Backward citations don't need recency filtering - they're historical
    references that should have accumulated citations.

    Args:
        seed_dois: DOIs to expand from
        min_citations: Minimum citation count for older forward citations
        recency_years: Years to consider "recent" (default: 3)

    Returns:
        Tuple of (candidate_papers, citation_edges)
    """
    forward_results: list[dict] = []
    backward_results: list[dict] = []
    citation_edges: list[CitationEdge] = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    # Calculate recency cutoff
    current_year = datetime.now(timezone.utc).year
    recent_cutoff = current_year - recency_years

    async def fetch_single_paper(
        seed_doi: str,
    ) -> tuple[list[dict], list[dict], list[CitationEdge]]:
        """Fetch both forward and backward citations for a single seed DOI."""
        async with semaphore:
            forward_papers = []
            backward_papers = []
            edges = []
            seen_dois: set[str] = set()

            # Phase 1: Recent forward citations (no min_citations)
            try:
                recent_forward = await get_forward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                    min_citations=0,  # No threshold for recent
                    from_year=recent_cutoff,
                )

                for work in recent_forward.results:
                    work_dict = (
                        work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    )
                    doi = work_dict.get("doi", "")
                    if doi:
                        doi_clean = (
                            doi.replace("https://doi.org/", "")
                            .replace("http://doi.org/", "")
                        )
                        if doi_clean not in seen_dois:
                            seen_dois.add(doi_clean)
                            forward_papers.append(work_dict)
                            edges.append(
                                CitationEdge(
                                    citing_doi=doi_clean,
                                    cited_doi=seed_doi,
                                    discovered_at=datetime.now(timezone.utc),
                                    edge_type="forward",
                                )
                            )

            except Exception as e:
                logger.warning(f"Failed to fetch recent forward citations for {seed_doi}: {e}")

            # Phase 2: Older forward citations (with min_citations)
            try:
                older_forward = await get_forward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                    min_citations=min_citations,
                )

                for work in older_forward.results:
                    work_dict = (
                        work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    )
                    doi = work_dict.get("doi", "")
                    if doi:
                        doi_clean = (
                            doi.replace("https://doi.org/", "")
                            .replace("http://doi.org/", "")
                        )
                        # Deduplicate - older query may include some recent papers
                        if doi_clean not in seen_dois:
                            seen_dois.add(doi_clean)
                            forward_papers.append(work_dict)
                            edges.append(
                                CitationEdge(
                                    citing_doi=doi_clean,
                                    cited_doi=seed_doi,
                                    discovered_at=datetime.now(timezone.utc),
                                    edge_type="forward",
                                )
                            )

            except Exception as e:
                logger.warning(f"Failed to fetch older forward citations for {seed_doi}: {e}")

            # Fetch backward citations
            try:
                backward_result = await get_backward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                )

                for work in backward_result.results:
                    work_dict = (
                        work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    )
                    backward_papers.append(work_dict)

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
                                discovered_at=datetime.now(timezone.utc),
                                edge_type="backward",
                            )
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to fetch backward citations for {seed_doi}: {e}"
                )

            logger.debug(
                f"Fetched {len(forward_papers)} forward, {len(backward_papers)} backward "
                f"citations for {seed_doi[:30]}..."
            )

            return forward_papers, backward_papers, edges

    # Fetch all citations in parallel
    tasks = [fetch_single_paper(doi) for doi in seed_dois]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Citation fetch task failed: {result}")
            continue
        forward, backward, edges = result
        forward_results.extend(forward)
        backward_results.extend(backward)
        citation_edges.extend(edges)

    all_results = forward_results + backward_results

    logger.debug(
        f"Fetched {len(forward_results)} forward, {len(backward_results)} backward "
        f"citations from {len(seed_dois)} seeds"
    )

    return all_results, citation_edges
