"""Main graph for academic literature review workflow.

Connects all phases into a complete literature review pipeline:
1. Discovery: Find initial papers via keyword search + citation network
2. Diffusion: Expand corpus via recursive citation network exploration
3. Processing: Acquire full text, extract summaries via document_processing
4. Clustering: Dual-strategy thematic organization (BERTopic + LLM + Opus)
5. Synthesis: Write coherent literature review with proper citations

Flow:
    START -> discovery_phase -> diffusion_phase -> processing_phase
          -> clustering_phase -> synthesis_phase -> END
"""

# Configure LangSmith tracing before other imports
from core.config import configure_langsmith

configure_langsmith()

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from workflows.research.subgraphs.academic_lit_review.state import (
    AcademicLitReviewState,
    LitReviewInput,
    QualitySettings,
    QUALITY_PRESETS,
    LitReviewDiffusionState,
)
from workflows.research.subgraphs.academic_lit_review.citation_graph import CitationGraph
from workflows.research.subgraphs.academic_lit_review.keyword_search import (
    run_keyword_search,
)
from workflows.research.subgraphs.academic_lit_review.citation_network import (
    run_citation_expansion,
)
from workflows.research.subgraphs.academic_lit_review.diffusion_engine import (
    run_diffusion,
)
from workflows.research.subgraphs.academic_lit_review.paper_processor import (
    run_paper_processing,
)
from workflows.research.subgraphs.academic_lit_review.clustering import (
    run_clustering,
)
from workflows.research.subgraphs.academic_lit_review.synthesis import (
    run_synthesis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Phase Node Functions
# =============================================================================


async def discovery_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 1: Discover seed papers through multiple strategies.

    Runs keyword search and initial citation network discovery in parallel.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])
    date_range = input_data.get("date_range")
    focus_areas = input_data.get("focus_areas")

    logger.info(f"Starting discovery phase for: {topic}")

    # Run keyword search
    keyword_result = await run_keyword_search(
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        date_range=date_range,
        focus_areas=focus_areas,
    )

    keyword_papers = keyword_result.get("discovered_papers", [])
    keyword_dois = keyword_result.get("keyword_dois", [])

    # Build initial paper corpus from keyword search
    paper_corpus = {}
    for paper in keyword_papers:
        doi = paper.get("doi")
        if doi:
            paper_corpus[doi] = paper

    # Run initial citation expansion on top keyword results
    top_seed_dois = keyword_dois[:10]  # Use top 10 as seeds

    if top_seed_dois:
        citation_result = await run_citation_expansion(
            seed_dois=top_seed_dois,
            topic=topic,
            research_questions=research_questions,
            quality_settings=quality_settings,
            existing_dois=set(paper_corpus.keys()),
        )

        citation_papers = citation_result.get("discovered_papers", [])
        citation_dois = citation_result.get("citation_dois", [])

        # Add citation-discovered papers to corpus
        for paper in citation_papers:
            doi = paper.get("doi")
            if doi and doi not in paper_corpus:
                paper_corpus[doi] = paper
    else:
        citation_dois = []

    logger.info(
        f"Discovery complete: {len(keyword_dois)} from keywords, "
        f"{len(citation_dois)} from citations, {len(paper_corpus)} total"
    )

    return {
        "keyword_papers": keyword_dois,
        "citation_papers": citation_dois,
        "paper_corpus": paper_corpus,
        "current_phase": "diffusion",
        "current_status": f"Discovery complete: {len(paper_corpus)} papers found",
    }


async def diffusion_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 2: Expand corpus through recursive citation diffusion.

    Iteratively explores citation network until saturation.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    paper_corpus = state.get("paper_corpus", {})

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    logger.info(f"Starting diffusion phase with {len(paper_corpus)} seed papers")

    # Use all current corpus papers as seeds for diffusion
    discovery_seeds = list(paper_corpus.keys())

    if not discovery_seeds:
        logger.warning("No seeds for diffusion, skipping phase")
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
            "current_phase": "processing",
            "current_status": "Diffusion skipped (no seeds)",
        }

    diffusion_result = await run_diffusion(
        discovery_seeds=discovery_seeds,
        paper_corpus=paper_corpus,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
    )

    final_corpus = diffusion_result.get("paper_corpus", paper_corpus)
    diffusion_state = diffusion_result.get("diffusion", {})
    saturation_reason = diffusion_result.get("saturation_reason", "Unknown")

    logger.info(
        f"Diffusion complete: {len(final_corpus)} papers in corpus. "
        f"Reason: {saturation_reason}"
    )

    return {
        "paper_corpus": final_corpus,
        "diffusion": diffusion_state,
        "papers_to_process": list(final_corpus.keys()),
        "current_phase": "processing",
        "current_status": f"Diffusion complete: {len(final_corpus)} papers ({saturation_reason})",
    }


async def processing_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 3: Process papers for full-text extraction and summarization.

    Uses document_processing workflow for PDF handling and summary extraction.
    """
    input_data = state["input"]
    paper_corpus = state.get("paper_corpus", {})
    quality_settings = state["quality_settings"]

    topic = input_data["topic"]

    logger.info(f"Starting processing phase for {len(paper_corpus)} papers")

    if not paper_corpus:
        logger.warning("No papers to process")
        return {
            "paper_summaries": {},
            "current_phase": "clustering",
            "current_status": "Processing skipped (no papers)",
        }

    # Convert corpus to list of PaperMetadata
    papers_to_process = list(paper_corpus.values())

    processing_result = await run_paper_processing(
        papers=papers_to_process,
        quality_settings=quality_settings,
        topic=topic,
    )

    paper_summaries = processing_result.get("paper_summaries", {})
    zotero_keys = processing_result.get("zotero_keys", {})
    es_ids = processing_result.get("elasticsearch_ids", {})
    processed = processing_result.get("processed_dois", [])
    failed = processing_result.get("failed_dois", [])

    logger.info(
        f"Processing complete: {len(processed)} successful, {len(failed)} failed"
    )

    return {
        "paper_summaries": paper_summaries,
        "zotero_keys": zotero_keys,
        "elasticsearch_ids": es_ids,
        "papers_processed": processed,
        "papers_failed": failed,
        "current_phase": "clustering",
        "current_status": f"Processing complete: {len(processed)} papers summarized",
    }


async def clustering_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 4: Cluster papers into thematic groups.

    Uses dual-strategy clustering (BERTopic + LLM) with Opus synthesis.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    paper_summaries = state.get("paper_summaries", {})

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    logger.info(f"Starting clustering phase for {len(paper_summaries)} papers")

    if not paper_summaries:
        logger.warning("No paper summaries to cluster")
        return {
            "clusters": [],
            "current_phase": "synthesis",
            "current_status": "Clustering skipped (no summaries)",
        }

    clustering_result = await run_clustering(
        paper_summaries=paper_summaries,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
    )

    clusters = clustering_result.get("final_clusters", [])
    cluster_analyses = clustering_result.get("cluster_analyses", [])
    bertopic_clusters = clustering_result.get("bertopic_clusters", [])
    llm_schema = clustering_result.get("llm_topic_schema")

    logger.info(f"Clustering complete: {len(clusters)} thematic clusters identified")

    return {
        "clusters": clusters,
        "bertopic_clusters": bertopic_clusters,
        "llm_topic_schema": llm_schema,
        # Store cluster analyses in state for synthesis
        "_cluster_analyses": cluster_analyses,
        "current_phase": "synthesis",
        "current_status": f"Clustering complete: {len(clusters)} themes identified",
    }


async def synthesis_phase_node(state: AcademicLitReviewState) -> dict[str, Any]:
    """Phase 5: Synthesize findings into coherent literature review.

    Writes thematic sections, integrates document, processes citations.
    """
    input_data = state["input"]
    quality_settings = state["quality_settings"]
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    zotero_keys = state.get("zotero_keys", {})

    # Retrieve cluster analyses from state
    cluster_analyses = state.get("_cluster_analyses", [])

    topic = input_data["topic"]
    research_questions = input_data.get("research_questions", [])

    logger.info(f"Starting synthesis phase with {len(clusters)} clusters")

    if not clusters or not paper_summaries:
        logger.warning("No clusters or summaries for synthesis")
        return {
            "final_review": "Literature review generation failed: no content available.",
            "current_phase": "complete",
            "current_status": "Synthesis skipped (no content)",
            "completed_at": datetime.utcnow(),
        }

    synthesis_result = await run_synthesis(
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        topic=topic,
        research_questions=research_questions,
        quality_settings=quality_settings,
        zotero_keys=zotero_keys,
    )

    final_review = synthesis_result.get("final_review", "")
    references = synthesis_result.get("references", [])
    quality_metrics = synthesis_result.get("quality_metrics", {})
    quality_passed = synthesis_result.get("quality_passed", False)
    prisma_docs = synthesis_result.get("prisma_documentation", "")

    logger.info(
        f"Synthesis complete: {quality_metrics.get('total_words', 0)} words, "
        f"{quality_metrics.get('unique_papers_cited', 0)} papers cited, "
        f"quality_passed={quality_passed}"
    )

    return {
        "final_review": final_review,
        "references": references,
        "prisma_documentation": prisma_docs,
        "section_drafts": synthesis_result.get("section_drafts", {}),
        "current_phase": "complete",
        "current_status": f"Complete: {quality_metrics.get('total_words', 0)} word review",
        "completed_at": datetime.utcnow(),
    }


# =============================================================================
# Main Graph Definition
# =============================================================================


def create_academic_lit_review_graph() -> StateGraph:
    """Create the main academic literature review workflow graph.

    Flow:
        START -> discovery -> diffusion -> processing
              -> clustering -> synthesis -> END
    """
    builder = StateGraph(AcademicLitReviewState)

    # Add phase nodes
    builder.add_node("discovery", discovery_phase_node)
    builder.add_node("diffusion", diffusion_phase_node)
    builder.add_node("processing", processing_phase_node)
    builder.add_node("clustering", clustering_phase_node)
    builder.add_node("synthesis", synthesis_phase_node)

    # Add edges (linear flow)
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "diffusion")
    builder.add_edge("diffusion", "processing")
    builder.add_edge("processing", "clustering")
    builder.add_edge("clustering", "synthesis")
    builder.add_edge("synthesis", END)

    return builder.compile()


# Export compiled graph
academic_lit_review_graph = create_academic_lit_review_graph()


# =============================================================================
# Main Entry Point
# =============================================================================


async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "standard",
    date_range: Optional[tuple[int, int]] = None,
    include_books: bool = True,
    focus_areas: Optional[list[str]] = None,
    exclude_terms: Optional[list[str]] = None,
    max_papers: Optional[int] = None,
) -> dict[str, Any]:
    """Run a complete academic literature review workflow.

    This is the main entry point for generating PhD-equivalent literature reviews.

    Args:
        topic: Research topic (e.g., "Large Language Models in Scientific Discovery")
        research_questions: List of specific questions to address
        quality: Quality tier - "quick", "standard", "comprehensive", "high_quality"
        date_range: Optional (start_year, end_year) filter
        include_books: Whether to include book sources (default: True)
        focus_areas: Optional specific areas to prioritize
        exclude_terms: Optional terms to filter out
        max_papers: Override default max papers for quality tier

    Returns:
        Dict containing:
        - final_review: Complete literature review text with citations
        - paper_corpus: All discovered papers
        - paper_summaries: Processed paper summaries
        - clusters: Thematic clusters
        - references: Formatted citations
        - prisma_documentation: Search methodology docs
        - quality_metrics: Review quality metrics
        - errors: Any errors encountered

    Example:
        result = await academic_lit_review(
            topic="Large Language Models in Scientific Discovery",
            research_questions=[
                "How are LLMs being used for hypothesis generation?",
                "What are the methodological challenges of using LLMs in research?",
            ],
            quality="high_quality",
            date_range=(2020, 2025),
        )

        # Access results
        print(f"Papers analyzed: {len(result['paper_corpus'])}")
        print(f"Review length: {len(result['final_review'].split())} words")

        # Save to file
        with open("literature_review.md", "w") as f:
            f.write(result['final_review'])
    """
    # Get quality settings
    if quality not in QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    quality_settings = QUALITY_PRESETS[quality].copy()

    # Override max_papers if specified
    if max_papers:
        quality_settings["max_papers"] = max_papers

    # Build input
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        date_range=date_range,
        include_books=include_books,
        focus_areas=focus_areas,
        exclude_terms=exclude_terms,
        max_papers=max_papers,
    )

    # Initialize state
    initial_state = AcademicLitReviewState(
        input=input_data,
        quality_settings=quality_settings,
        keyword_papers=[],
        citation_papers=[],
        expert_papers=[],
        book_dois=[],
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
        paper_corpus={},
        paper_summaries={},
        citation_edges=[],
        paper_nodes={},
        papers_to_process=[],
        papers_processed=[],
        papers_failed=[],
        bertopic_clusters=None,
        llm_topic_schema=None,
        clusters=[],
        section_drafts={},
        final_review=None,
        references=[],
        prisma_documentation=None,
        elasticsearch_ids={},
        zotero_keys={},
        started_at=datetime.utcnow(),
        completed_at=None,
        current_phase="discovery",
        current_status="Starting literature review",
        langsmith_run_id=str(uuid.uuid4()),
        errors=[],
    )

    logger.info(f"Starting academic literature review: {topic}")
    logger.info(f"Quality: {quality}, Max papers: {quality_settings['max_papers']}")
    logger.info(f"LangSmith run ID: {initial_state['langsmith_run_id']}")

    try:
        result = await academic_lit_review_graph.ainvoke(initial_state)

        return {
            "final_review": result.get("final_review", ""),
            "paper_corpus": result.get("paper_corpus", {}),
            "paper_summaries": result.get("paper_summaries", {}),
            "clusters": result.get("clusters", []),
            "references": result.get("references", []),
            "citation_keys": list(result.get("zotero_keys", {}).values()),
            "zotero_keys": result.get("zotero_keys", {}),
            "elasticsearch_ids": result.get("elasticsearch_ids", {}),
            "prisma_documentation": result.get("prisma_documentation", ""),
            "diffusion": result.get("diffusion", {}),
            "quality_metrics": result.get("section_drafts", {}).get("quality_metrics"),
            "started_at": initial_state["started_at"],
            "completed_at": result.get("completed_at"),
            "langsmith_run_id": initial_state["langsmith_run_id"],
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error(f"Literature review failed: {e}")
        return {
            "final_review": f"Literature review generation failed: {e}",
            "paper_corpus": {},
            "paper_summaries": {},
            "clusters": [],
            "references": [],
            "langsmith_run_id": initial_state["langsmith_run_id"],
            "errors": [{"phase": "unknown", "error": str(e)}],
        }
