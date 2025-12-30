"""Diffusion engine subgraph for iterative corpus expansion through citation network.

Implements multi-stage citation diffusion with two-stage relevance filtering:
1. Co-citation analysis (automatic inclusion for highly co-cited papers)
2. LLM-based relevance scoring for remaining candidates

Flow:
    START -> initialize -> select_seeds -> citation_expansion -> cocitation_check
          -> llm_scoring -> update_corpus -> check_saturation
          -> (continue: select_seeds OR finalize: finalize -> END)
"""

import asyncio
import logging
from datetime import datetime
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
    DiffusionStage,
    LitReviewDiffusionState,
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.research.subgraphs.academic_lit_review.citation_graph import CitationGraph
from workflows.research.subgraphs.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    batch_score_relevance,
)
from workflows.shared.llm_utils import ModelTier

logger = logging.getLogger(__name__)

# Constants
MAX_CITATIONS_PER_PAPER = 30
MAX_CONCURRENT_FETCHES = 5
COCITATION_THRESHOLD = 3  # Papers co-cited with 3+ corpus papers auto-include


# =============================================================================
# State Definition
# =============================================================================


class DiffusionEngineState(TypedDict):
    """State for diffusion engine subgraph."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    discovery_seeds: list[str]  # DOIs from discovery phase

    # Citation graph (accumulated)
    citation_graph: CitationGraph

    # Paper corpus (accumulated)
    paper_corpus: dict[str, PaperMetadata]

    # Diffusion tracking
    diffusion: LitReviewDiffusionState

    # Current stage working state
    current_stage_seeds: list[str]
    current_stage_candidates: list[PaperMetadata]
    current_stage_relevant: list[str]
    current_stage_rejected: list[str]
    new_citation_edges: list[CitationEdge]
    cocitation_included: list[str]

    # Output
    final_corpus_dois: list[str]
    saturation_reason: Optional[str]


# =============================================================================
# Helper Functions
# =============================================================================


async def fetch_citations_raw(
    seed_dois: list[str],
    min_citations: int = 10,
) -> tuple[list[dict], list[CitationEdge]]:
    """Fetch forward and backward citations without relevance filtering.

    This allows the two-stage relevance filter (co-citation + LLM) to work
    on the full candidate set.

    Args:
        seed_dois: DOIs to expand from
        min_citations: Minimum citation count for forward citations

    Returns:
        Tuple of (candidate_papers, citation_edges)
    """
    forward_results: list[dict] = []
    backward_results: list[dict] = []
    citation_edges: list[CitationEdge] = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def fetch_single_paper(seed_doi: str) -> tuple[list[dict], list[dict], list[CitationEdge]]:
        """Fetch both forward and backward citations for a single seed DOI."""
        async with semaphore:
            forward_papers = []
            backward_papers = []
            edges = []

            # Fetch forward citations
            try:
                forward_result = await get_forward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                    min_citations=min_citations,
                )

                for work in forward_result.results:
                    work_dict = work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    forward_papers.append(work_dict)

                    if work_dict.get("doi"):
                        citing_doi = work_dict["doi"].replace("https://doi.org/", "").replace("http://doi.org/", "")
                        edges.append(
                            CitationEdge(
                                citing_doi=citing_doi,
                                cited_doi=seed_doi,
                                discovered_at=datetime.utcnow(),
                                edge_type="forward",
                            )
                        )

            except Exception as e:
                logger.warning(f"Failed to fetch forward citations for {seed_doi}: {e}")

            # Fetch backward citations
            try:
                backward_result = await get_backward_citations(
                    work_id=seed_doi,
                    limit=MAX_CITATIONS_PER_PAPER,
                )

                for work in backward_result.results:
                    work_dict = work.model_dump() if hasattr(work, "model_dump") else dict(work)
                    backward_papers.append(work_dict)

                    if work_dict.get("doi"):
                        cited_doi = work_dict["doi"].replace("https://doi.org/", "").replace("http://doi.org/", "")
                        edges.append(
                            CitationEdge(
                                citing_doi=seed_doi,
                                cited_doi=cited_doi,
                                discovered_at=datetime.utcnow(),
                                edge_type="backward",
                            )
                        )

            except Exception as e:
                logger.warning(f"Failed to fetch backward citations for {seed_doi}: {e}")

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

    logger.info(
        f"Fetched {len(forward_results)} forward, {len(backward_results)} backward "
        f"citations from {len(seed_dois)} seeds"
    )

    return all_results, citation_edges


# =============================================================================
# Node Functions
# =============================================================================


async def initialize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Initialize diffusion state from discovery seeds."""
    discovery_seeds = state.get("discovery_seeds", [])
    quality_settings = state["quality_settings"]

    if not discovery_seeds:
        logger.warning("No discovery seeds provided for diffusion")
        return {
            "diffusion": LitReviewDiffusionState(
                current_stage=0,
                max_stages=quality_settings["max_stages"],
                stages=[],
                saturation_threshold=quality_settings["saturation_threshold"],
                is_saturated=True,
                consecutive_low_coverage=0,
                total_papers_discovered=0,
                total_papers_relevant=0,
                total_papers_rejected=0,
            ),
            "saturation_reason": "No discovery seeds provided",
        }

    # Initialize diffusion tracking
    diffusion_state = LitReviewDiffusionState(
        current_stage=0,
        max_stages=quality_settings["max_stages"],
        stages=[],
        saturation_threshold=quality_settings["saturation_threshold"],
        is_saturated=False,
        consecutive_low_coverage=0,
        total_papers_discovered=len(discovery_seeds),
        total_papers_relevant=len(discovery_seeds),
        total_papers_rejected=0,
    )

    logger.info(
        f"Initialized diffusion with {len(discovery_seeds)} seeds, "
        f"max_stages={quality_settings['max_stages']}, "
        f"saturation_threshold={quality_settings['saturation_threshold']}"
    )

    return {
        "diffusion": diffusion_state,
        "current_stage_seeds": [],
        "current_stage_candidates": [],
        "current_stage_relevant": [],
        "current_stage_rejected": [],
        "new_citation_edges": [],
        "cocitation_included": [],
    }


async def select_expansion_seeds(state: DiffusionEngineState) -> dict[str, Any]:
    """Select papers to expand from using citation graph analysis."""
    citation_graph = state.get("citation_graph")
    diffusion = state["diffusion"]
    quality_settings = state["quality_settings"]

    if not citation_graph or citation_graph.node_count == 0:
        logger.warning("No papers in citation graph to select seeds from")
        return {
            "current_stage_seeds": [],
            "diffusion": {**diffusion, "is_saturated": True},
            "saturation_reason": "No papers in citation graph",
        }

    # Increment stage
    new_stage = diffusion["current_stage"] + 1

    # Use citation graph to get prioritized expansion candidates
    max_seeds = min(20, quality_settings.get("max_papers", 100) // 10)
    seed_dois = citation_graph.get_expansion_candidates(
        max_papers=max_seeds,
        prioritize_recent=True,
    )

    if not seed_dois:
        logger.info("No expansion candidates found, saturation reached")
        return {
            "current_stage_seeds": [],
            "diffusion": {**diffusion, "is_saturated": True},
            "saturation_reason": "No expansion candidates available",
        }

    # Create new stage record
    new_stage_record = DiffusionStage(
        stage_number=new_stage,
        seed_papers=seed_dois,
        forward_papers_found=0,
        backward_papers_found=0,
        new_relevant=[],
        new_rejected=[],
        coverage_delta=0.0,
        started_at=datetime.utcnow(),
        completed_at=None,
    )

    updated_diffusion = {
        **diffusion,
        "current_stage": new_stage,
        "stages": diffusion["stages"] + [new_stage_record],
    }

    logger.info(f"Stage {new_stage}: Selected {len(seed_dois)} expansion seeds")

    return {
        "current_stage_seeds": seed_dois,
        "diffusion": updated_diffusion,
    }


async def run_citation_expansion_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Fetch forward and backward citations for selected seeds."""
    seed_dois = state.get("current_stage_seeds", [])
    quality_settings = state["quality_settings"]
    existing_dois = set(state.get("paper_corpus", {}).keys())

    if not seed_dois:
        logger.warning("No seeds to expand from")
        return {
            "current_stage_candidates": [],
            "new_citation_edges": [],
        }

    min_citations = quality_settings.get("min_citations_filter", 10)

    # Fetch all citations without relevance filtering
    raw_results, citation_edges = await fetch_citations_raw(
        seed_dois=seed_dois,
        min_citations=min_citations,
    )

    # Convert to PaperMetadata
    diffusion = state["diffusion"]
    current_stage = diffusion["current_stage"]

    candidates = []
    for result in raw_results:
        paper = convert_to_paper_metadata(
            work=result,
            discovery_stage=current_stage,
            discovery_method="diffusion",
        )
        if paper:
            candidates.append(paper)

    # Deduplicate and remove existing corpus papers
    candidates = deduplicate_papers(candidates, existing_dois)

    logger.info(
        f"Stage {current_stage}: Fetched {len(raw_results)} citations, "
        f"{len(candidates)} unique candidates after deduplication"
    )

    return {
        "current_stage_candidates": candidates,
        "new_citation_edges": citation_edges,
    }


async def check_cocitation_relevance_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Two-stage relevance: Stage 1 - auto-include papers co-cited with 3+ corpus papers."""
    candidates = state.get("current_stage_candidates", [])
    citation_graph = state.get("citation_graph")
    citation_edges = state.get("new_citation_edges", [])
    corpus_dois = set(state.get("paper_corpus", {}).keys())

    if not candidates or not citation_graph:
        return {
            "cocitation_included": [],
            "current_stage_candidates": candidates,
        }

    # Temporarily add new edges to graph for co-citation analysis
    temp_graph = CitationGraph()

    # Add existing nodes
    for doi, metadata in state.get("paper_corpus", {}).items():
        temp_graph.add_paper(doi, metadata)

    # Add candidate nodes
    for candidate in candidates:
        doi = candidate.get("doi")
        if doi:
            temp_graph.add_paper(doi, candidate)

    # Add all edges (existing + new)
    for edge in citation_edges:
        temp_graph.add_citation(
            citing_doi=edge["citing_doi"],
            cited_doi=edge["cited_doi"],
            edge_type=edge["edge_type"],
        )

    # Check co-citation for each candidate
    cocitation_included = []
    remaining_candidates = []

    for candidate in candidates:
        doi = candidate.get("doi")
        if not doi:
            continue

        # Check if this candidate is co-cited with corpus papers
        is_cocited = temp_graph.get_cocitation_candidates(
            paper_doi=doi,
            corpus_dois=corpus_dois,
            threshold=COCITATION_THRESHOLD,
        )

        if is_cocited:
            cocitation_included.append(doi)
            logger.debug(f"Auto-included via co-citation: {candidate.get('title', 'Unknown')[:50]}")
        else:
            remaining_candidates.append(candidate)

    logger.info(
        f"Co-citation filter: {len(cocitation_included)} auto-included, "
        f"{len(remaining_candidates)} require LLM scoring"
    )

    return {
        "cocitation_included": cocitation_included,
        "current_stage_candidates": remaining_candidates,
    }


async def score_remaining_relevance_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Two-stage relevance: Stage 2 - LLM scoring for remaining candidates."""
    candidates = state.get("current_stage_candidates", [])
    input_data = state["input"]
    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    if not candidates:
        logger.info("No candidates remaining for LLM relevance scoring")
        return {
            "current_stage_relevant": [],
            "current_stage_rejected": [],
        }

    # Score relevance with LLM
    relevant, rejected = await batch_score_relevance(
        papers=candidates,
        topic=topic,
        research_questions=research_questions,
        threshold=0.6,
        tier=ModelTier.HAIKU,
        max_concurrent=10,
    )

    # Extract DOIs
    relevant_dois = [p.get("doi") for p in relevant if p.get("doi")]
    rejected_dois = [p.get("doi") for p in rejected if p.get("doi")]

    logger.info(
        f"LLM relevance scoring: {len(relevant_dois)} relevant, {len(rejected_dois)} rejected"
    )

    return {
        "current_stage_relevant": relevant_dois,
        "current_stage_rejected": rejected_dois,
    }


async def update_corpus_and_graph(state: DiffusionEngineState) -> dict[str, Any]:
    """Add newly relevant papers to corpus and update citation graph."""
    cocitation_included = state.get("cocitation_included", [])
    llm_relevant = state.get("current_stage_relevant", [])
    llm_rejected = state.get("current_stage_rejected", [])
    candidates = state.get("current_stage_candidates", [])
    citation_edges = state.get("new_citation_edges", [])
    citation_graph = state.get("citation_graph")
    paper_corpus = state.get("paper_corpus", {})
    diffusion = state["diffusion"]

    # Combine all relevant DOIs
    all_relevant_dois = set(cocitation_included) | set(llm_relevant)

    # Build lookup for candidates
    candidate_lookup = {p.get("doi"): p for p in candidates if p.get("doi")}

    # Also include co-citation candidates (need to rebuild from all candidates)
    all_candidates_lookup = {
        p.get("doi"): p
        for p in state.get("current_stage_candidates", [])
        if p.get("doi")
    }

    # Merge with original candidates that might have been filtered
    for doi in cocitation_included:
        if doi not in candidate_lookup and doi not in all_candidates_lookup:
            # This shouldn't happen, but handle gracefully
            logger.warning(f"Co-cited paper {doi} not found in candidates")

    # Add relevant papers to corpus
    new_corpus_papers = {}
    for doi in all_relevant_dois:
        # Try to find in candidate lookup first, then all candidates
        paper = candidate_lookup.get(doi) or all_candidates_lookup.get(doi)
        if paper:
            new_corpus_papers[doi] = paper

    # Add papers to citation graph
    if citation_graph:
        for doi, paper in new_corpus_papers.items():
            citation_graph.add_paper(doi, paper)

        # Add citation edges
        for edge in citation_edges:
            citation_graph.add_citation(
                citing_doi=edge["citing_doi"],
                cited_doi=edge["cited_doi"],
                edge_type=edge["edge_type"],
            )

    # Calculate coverage delta
    total_candidates = len(cocitation_included) + len(llm_relevant) + len(llm_rejected)
    coverage_delta = len(all_relevant_dois) / total_candidates if total_candidates > 0 else 0.0

    # Update stage record
    current_stage = diffusion["current_stage"]
    stages = diffusion["stages"]
    if stages and stages[-1]["stage_number"] == current_stage:
        stages[-1]["new_relevant"] = list(all_relevant_dois)
        stages[-1]["new_rejected"] = llm_rejected
        stages[-1]["coverage_delta"] = coverage_delta
        stages[-1]["completed_at"] = datetime.utcnow()

    # Update diffusion tracking
    new_consecutive_low = (
        diffusion["consecutive_low_coverage"] + 1
        if coverage_delta < diffusion["saturation_threshold"]
        else 0
    )

    updated_diffusion = {
        **diffusion,
        "stages": stages,
        "consecutive_low_coverage": new_consecutive_low,
        "total_papers_discovered": diffusion["total_papers_discovered"] + total_candidates,
        "total_papers_relevant": diffusion["total_papers_relevant"] + len(all_relevant_dois),
        "total_papers_rejected": diffusion["total_papers_rejected"] + len(llm_rejected),
    }

    # Merge with existing corpus
    updated_corpus = {**paper_corpus, **new_corpus_papers}

    logger.info(
        f"Stage {current_stage}: Added {len(new_corpus_papers)} papers to corpus "
        f"(coverage_delta={coverage_delta:.2f}). Total corpus: {len(updated_corpus)} papers"
    )

    return {
        "paper_corpus": updated_corpus,
        "citation_graph": citation_graph,
        "diffusion": updated_diffusion,
        # Reset working state for next iteration
        "current_stage_candidates": [],
        "current_stage_relevant": [],
        "current_stage_rejected": [],
        "new_citation_edges": [],
        "cocitation_included": [],
    }


async def check_saturation_node(state: DiffusionEngineState) -> dict[str, Any]:
    """Check if diffusion should stop based on saturation conditions."""
    diffusion = state["diffusion"]
    quality_settings = state["quality_settings"]
    paper_corpus = state.get("paper_corpus", {})

    # Check stopping conditions
    saturation_reason = None

    # 1. Max stages reached
    if diffusion["current_stage"] >= diffusion["max_stages"]:
        saturation_reason = f"Reached maximum stages ({diffusion['max_stages']})"

    # 2. Max papers reached
    elif len(paper_corpus) >= quality_settings["max_papers"]:
        saturation_reason = f"Reached maximum papers ({quality_settings['max_papers']})"

    # 3. Consecutive low coverage (2 stages with delta < threshold)
    elif diffusion["consecutive_low_coverage"] >= 2:
        saturation_reason = (
            f"Low coverage for {diffusion['consecutive_low_coverage']} consecutive stages "
            f"(threshold={diffusion['saturation_threshold']})"
        )

    if saturation_reason:
        logger.info(f"Diffusion saturation: {saturation_reason}")
        return {
            "diffusion": {**diffusion, "is_saturated": True},
            "saturation_reason": saturation_reason,
        }
    else:
        logger.info(
            f"Continuing diffusion: stage {diffusion['current_stage']}/{diffusion['max_stages']}, "
            f"corpus size {len(paper_corpus)}/{quality_settings['max_papers']}"
        )
        return {
            "diffusion": diffusion,
        }


async def finalize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Finalize diffusion and return final corpus DOIs."""
    paper_corpus = state.get("paper_corpus", {})
    diffusion = state["diffusion"]
    saturation_reason = state.get("saturation_reason", "Unknown")

    final_dois = list(paper_corpus.keys())

    logger.info(
        f"Diffusion complete: {len(final_dois)} papers in final corpus. "
        f"Reason: {saturation_reason}"
    )

    return {
        "final_corpus_dois": final_dois,
    }


# =============================================================================
# Conditional Routing
# =============================================================================


def should_continue_diffusion(state: DiffusionEngineState) -> str:
    """Determine if diffusion should continue or finalize."""
    diffusion = state.get("diffusion", {})

    if diffusion.get("is_saturated", False):
        return "finalize"
    return "continue"


# =============================================================================
# Subgraph Definition
# =============================================================================


def create_diffusion_engine_subgraph() -> StateGraph:
    """Create the diffusion engine subgraph.

    Flow:
        START -> initialize -> select_seeds -> citation_expansion -> cocitation_check
              -> llm_scoring -> update_corpus -> check_saturation
              -> (continue: select_seeds OR finalize: finalize -> END)
    """
    builder = StateGraph(DiffusionEngineState)

    # Add nodes
    builder.add_node("initialize", initialize_diffusion)
    builder.add_node("select_seeds", select_expansion_seeds)
    builder.add_node("citation_expansion", run_citation_expansion_node)
    builder.add_node("cocitation_check", check_cocitation_relevance_node)
    builder.add_node("llm_scoring", score_remaining_relevance_node)
    builder.add_node("update_corpus", update_corpus_and_graph)
    builder.add_node("check_saturation", check_saturation_node)
    builder.add_node("finalize", finalize_diffusion)

    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "select_seeds")
    builder.add_edge("select_seeds", "citation_expansion")
    builder.add_edge("citation_expansion", "cocitation_check")
    builder.add_edge("cocitation_check", "llm_scoring")
    builder.add_edge("llm_scoring", "update_corpus")
    builder.add_edge("update_corpus", "check_saturation")

    # Conditional edge: continue or finalize
    builder.add_conditional_edges(
        "check_saturation",
        should_continue_diffusion,
        {
            "continue": "select_seeds",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    return builder.compile()


# Export compiled subgraph
diffusion_engine_subgraph = create_diffusion_engine_subgraph()


# =============================================================================
# Convenience Function
# =============================================================================


async def run_diffusion(
    discovery_seeds: list[str],
    paper_corpus: dict[str, PaperMetadata],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
) -> dict[str, Any]:
    """Run diffusion engine as a standalone operation.

    Args:
        discovery_seeds: DOIs from discovery phase to seed diffusion
        paper_corpus: Initial paper corpus (from discovery)
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings

    Returns:
        Dict with final_corpus_dois, paper_corpus, citation_graph, diffusion state
    """
    # Initialize citation graph with corpus papers
    citation_graph = CitationGraph()
    for doi, metadata in paper_corpus.items():
        citation_graph.add_paper(doi, metadata)

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

    initial_state = DiffusionEngineState(
        input=input_data,
        quality_settings=quality_settings,
        discovery_seeds=discovery_seeds,
        citation_graph=citation_graph,
        paper_corpus=paper_corpus,
        diffusion=LitReviewDiffusionState(
            current_stage=0,
            max_stages=quality_settings["max_stages"],
            stages=[],
            saturation_threshold=quality_settings["saturation_threshold"],
            is_saturated=False,
            consecutive_low_coverage=0,
            total_papers_discovered=0,
            total_papers_relevant=0,
            total_papers_rejected=0,
        ),
        current_stage_seeds=[],
        current_stage_candidates=[],
        current_stage_relevant=[],
        current_stage_rejected=[],
        new_citation_edges=[],
        cocitation_included=[],
        final_corpus_dois=[],
        saturation_reason=None,
    )

    result = await diffusion_engine_subgraph.ainvoke(initial_state)

    return {
        "final_corpus_dois": result.get("final_corpus_dois", []),
        "paper_corpus": result.get("paper_corpus", {}),
        "citation_graph": result.get("citation_graph"),
        "diffusion": result.get("diffusion"),
        "saturation_reason": result.get("saturation_reason"),
    }
