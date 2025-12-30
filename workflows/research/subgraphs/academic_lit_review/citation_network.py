"""Citation network subgraph for literature discovery via snowballing.

Fetches forward citations (papers that cite seeds) and backward citations
(papers cited by seeds) to expand the paper corpus through citation network
exploration.

Flow:
    START -> fetch_forward_citations -> fetch_backward_citations -> merge_and_filter -> END
"""

import asyncio
import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langchain_tools.openalex import (
    get_forward_citations,
    get_backward_citations,
    OpenAlexWork,
)
from workflows.research.subgraphs.academic_lit_review.state import (
    CitationEdge,
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.research.subgraphs.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    batch_score_relevance,
)
from workflows.shared.llm_utils import ModelTier

logger = logging.getLogger(__name__)

# Constants
MAX_CITATIONS_PER_PAPER = 30  # Limit per paper to avoid explosion
MAX_CONCURRENT_FETCHES = 5  # Concurrent OpenAlex API calls


# =============================================================================
# State Definition
# =============================================================================


class CitationNetworkState(TypedDict):
    """State for citation-based discovery subgraph."""

    # Input (from parent)
    input: LitReviewInput
    quality_settings: QualitySettings
    seed_dois: list[str]  # DOIs to expand from
    existing_dois: set[str]  # DOIs already in corpus (for deduplication)

    # Internal state
    forward_results: list[dict]  # Papers citing seeds
    backward_results: list[dict]  # Papers cited by seeds
    citation_edges: list[CitationEdge]  # Discovered edges

    # Output
    discovered_papers: list[PaperMetadata]  # After filtering
    rejected_papers: list[PaperMetadata]  # Papers that didn't pass filter
    discovered_dois: list[str]  # DOIs of discovered papers
    new_edges: list[CitationEdge]  # Citation edges to add to graph


# =============================================================================
# Node Functions
# =============================================================================


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

    async def fetch_single_forward(seed_doi: str) -> tuple[list[dict], list[CitationEdge]]:
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
                    # Convert OpenAlexWork to dict if needed
                    work_dict = work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    papers.append(work_dict)

                    # Record citation edge: citing_paper -> seed_paper
                    if work_dict.get("doi"):
                        citing_doi = work_dict["doi"].replace("https://doi.org/", "").replace("http://doi.org/", "")
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

    # Fetch all forward citations in parallel
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

    async def fetch_single_backward(seed_doi: str) -> tuple[list[dict], list[CitationEdge]]:
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
                    work_dict = work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    papers.append(work_dict)

                    # Record citation edge: seed_paper -> cited_paper
                    if work_dict.get("doi"):
                        cited_doi = work_dict["doi"].replace("https://doi.org/", "").replace("http://doi.org/", "")
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
                logger.warning(f"Failed to fetch backward citations for {seed_doi}: {e}")
                return [], []

    # Fetch all backward citations in parallel
    tasks = [fetch_single_backward(doi) for doi in seed_dois]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Backward citation task failed: {result}")
            continue
        papers, edges = result
        backward_results.extend(papers)
        new_edges.extend(edges)

    # Combine with existing edges
    all_edges = existing_edges + new_edges

    logger.info(
        f"Fetched {len(backward_results)} backward citations from {len(seed_dois)} seeds"
    )

    return {
        "backward_results": backward_results,
        "citation_edges": all_edges,
    }


async def merge_and_filter_node(state: CitationNetworkState) -> dict[str, Any]:
    """Merge forward/backward results, deduplicate, and filter by relevance.

    Combines all discovered papers, removes duplicates and papers already
    in corpus, then scores relevance to filter down to useful additions.
    """
    forward_results = state.get("forward_results", [])
    backward_results = state.get("backward_results", [])
    existing_dois = state.get("existing_dois", set())
    input_data = state["input"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    citation_edges = state.get("citation_edges", [])

    # Merge all results
    all_results = forward_results + backward_results

    if not all_results:
        logger.warning("No citation results to merge and filter")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "discovered_dois": [],
            "new_edges": citation_edges,
        }

    # Convert to PaperMetadata
    papers = []
    for result in all_results:
        # Determine discovery method based on whether result came from forward or backward
        # (This is a simplification - in practice we'd track this more precisely)
        discovery_method = "citation"

        paper = convert_to_paper_metadata(
            work=result,
            discovery_stage=1,  # Citation discovery is stage 1+
            discovery_method=discovery_method,
        )
        if paper:
            papers.append(paper)

    # Deduplicate and remove existing corpus papers
    papers = deduplicate_papers(papers, existing_dois)

    if not papers:
        logger.warning("No new papers after deduplication")
        return {
            "discovered_papers": [],
            "rejected_papers": [],
            "discovered_dois": [],
            "new_edges": citation_edges,
        }

    logger.info(f"Merged {len(all_results)} raw results to {len(papers)} unique new papers")

    # Score relevance
    relevant, rejected = await batch_score_relevance(
        papers=papers,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        tier=ModelTier.HAIKU,
        max_concurrent=10,
    )

    # Extract DOIs
    discovered_dois = [p.get("doi") for p in relevant if p.get("doi")]

    # Filter edges to only include those with at least one endpoint in relevant set
    discovered_doi_set = set(discovered_dois)
    seed_dois_set = set(state.get("seed_dois", []))
    valid_dois = discovered_doi_set | seed_dois_set

    filtered_edges = [
        edge for edge in citation_edges
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


# =============================================================================
# Subgraph Definition
# =============================================================================


def create_citation_network_subgraph() -> StateGraph:
    """Create the citation network discovery subgraph.

    Flow:
        START -> fetch_forward -> fetch_backward -> merge_and_filter -> END
    """
    builder = StateGraph(CitationNetworkState)

    # Add nodes
    builder.add_node("fetch_forward_citations", fetch_forward_citations_node)
    builder.add_node("fetch_backward_citations", fetch_backward_citations_node)
    builder.add_node("merge_and_filter", merge_and_filter_node)

    # Add edges
    builder.add_edge(START, "fetch_forward_citations")
    builder.add_edge("fetch_forward_citations", "fetch_backward_citations")
    builder.add_edge("fetch_backward_citations", "merge_and_filter")
    builder.add_edge("merge_and_filter", END)

    return builder.compile()


# Export compiled subgraph
citation_network_subgraph = create_citation_network_subgraph()


# =============================================================================
# Convenience Function
# =============================================================================


async def run_citation_expansion(
    seed_dois: list[str],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    existing_dois: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Run citation network expansion as a standalone operation.

    Args:
        seed_dois: DOIs to expand citations from
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        existing_dois: Optional set of DOIs already in corpus

    Returns:
        Dict with discovered_papers, rejected_papers, discovered_dois, new_edges
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
        include_books=False,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
    )

    initial_state = CitationNetworkState(
        input=input_data,
        quality_settings=quality_settings,
        seed_dois=seed_dois,
        existing_dois=existing_dois or set(),
        forward_results=[],
        backward_results=[],
        citation_edges=[],
        discovered_papers=[],
        rejected_papers=[],
        discovered_dois=[],
        new_edges=[],
    )

    result = await citation_network_subgraph.ainvoke(initial_state)
    return {
        "discovered_papers": result.get("discovered_papers", []),
        "rejected_papers": result.get("rejected_papers", []),
        "discovered_dois": result.get("discovered_dois", []),
        "new_edges": result.get("new_edges", []),
    }
