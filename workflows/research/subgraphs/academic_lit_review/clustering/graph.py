"""LangGraph subgraph definitions for clustering workflow."""

import asyncio
import logging
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from workflows.research.subgraphs.academic_lit_review.state import LitReviewInput, QualitySettings

from .analysis import ClusterAnalysis, per_cluster_analysis_node
from .bertopic_clustering import run_bertopic_clustering_node
from .formatters import prepare_document_for_clustering
from .llm_clustering import run_llm_clustering_node
from .synthesis import synthesize_clusters_node

logger = logging.getLogger(__name__)


class ClusteringState(TypedDict):
    """State for dual-strategy thematic clustering subgraph."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    paper_summaries: dict

    # Prepared documents for clustering
    document_texts: list[str]  # Formatted texts for BERTopic
    document_dois: list[str]  # DOIs in same order as document_texts

    # Parallel clustering results
    bertopic_clusters: list | None
    bertopic_error: str | None
    llm_topic_schema: dict | None
    llm_error: str | None

    # Synthesized result
    final_clusters: list
    cluster_labels: dict[str, int]  # DOI -> cluster_id

    # Per-cluster analysis
    cluster_analyses: list[ClusterAnalysis]


async def prepare_documents_node(state: ClusteringState) -> dict[str, Any]:
    """Prepare paper summaries as documents for clustering."""
    paper_summaries = state.get("paper_summaries", {})

    if not paper_summaries:
        logger.warning("No paper summaries to cluster")
        return {
            "document_texts": [],
            "document_dois": [],
        }

    document_texts = []
    document_dois = []

    for doi, summary in paper_summaries.items():
        doc_text = prepare_document_for_clustering(summary)
        document_texts.append(doc_text)
        document_dois.append(doi)

    logger.info(f"Prepared {len(document_texts)} documents for clustering")

    return {
        "document_texts": document_texts,
        "document_dois": document_dois,
    }


def create_clustering_subgraph() -> StateGraph:
    """Create the dual-strategy clustering subgraph.

    Flow:
        START -> prepare_documents -> parallel[bertopic, llm]
              -> synthesize_clusters -> per_cluster_analysis -> END

    Note: LangGraph doesn't support true parallel execution, so we
    run clustering methods sequentially but could be parallelized
    with asyncio in a custom node.
    """
    builder = StateGraph(ClusteringState)

    # Add nodes
    builder.add_node("prepare_documents", prepare_documents_node)
    builder.add_node("bertopic_clustering", run_bertopic_clustering_node)
    builder.add_node("llm_clustering", run_llm_clustering_node)
    builder.add_node("synthesize_clusters", synthesize_clusters_node)
    builder.add_node("per_cluster_analysis", per_cluster_analysis_node)

    # Add edges
    # Sequential for now - could be parallelized with a custom parallel node
    builder.add_edge(START, "prepare_documents")
    builder.add_edge("prepare_documents", "bertopic_clustering")
    builder.add_edge("bertopic_clustering", "llm_clustering")
    builder.add_edge("llm_clustering", "synthesize_clusters")
    builder.add_edge("synthesize_clusters", "per_cluster_analysis")
    builder.add_edge("per_cluster_analysis", END)

    return builder.compile()


def create_parallel_clustering_subgraph() -> StateGraph:
    """Create clustering subgraph with parallel BERTopic and LLM clustering.

    Uses a custom node to run both clustering methods concurrently.
    """
    async def parallel_clustering_node(state: ClusteringState) -> dict[str, Any]:
        """Run BERTopic and LLM clustering in parallel."""
        bertopic_task = asyncio.create_task(run_bertopic_clustering_node(state))
        llm_task = asyncio.create_task(run_llm_clustering_node(state))

        bertopic_result, llm_result = await asyncio.gather(
            bertopic_task, llm_task, return_exceptions=True
        )

        result = {}

        if isinstance(bertopic_result, Exception):
            logger.error(f"BERTopic clustering failed: {bertopic_result}")
            result["bertopic_clusters"] = []
            result["bertopic_error"] = str(bertopic_result)
        else:
            result.update(bertopic_result)

        if isinstance(llm_result, Exception):
            logger.error(f"LLM clustering failed: {llm_result}")
            result["llm_topic_schema"] = None
            result["llm_error"] = str(llm_result)
        else:
            result.update(llm_result)

        return result

    builder = StateGraph(ClusteringState)

    builder.add_node("prepare_documents", prepare_documents_node)
    builder.add_node("parallel_clustering", parallel_clustering_node)
    builder.add_node("synthesize_clusters", synthesize_clusters_node)
    builder.add_node("per_cluster_analysis", per_cluster_analysis_node)

    builder.add_edge(START, "prepare_documents")
    builder.add_edge("prepare_documents", "parallel_clustering")
    builder.add_edge("parallel_clustering", "synthesize_clusters")
    builder.add_edge("synthesize_clusters", "per_cluster_analysis")
    builder.add_edge("per_cluster_analysis", END)

    return builder.compile()


# Export the parallel version as default
clustering_subgraph = create_parallel_clustering_subgraph()
