"""Convenience API for synthesis subgraph."""

from typing import Any, Optional

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.research.academic_lit_review.clustering import ClusterAnalysis
from workflows.shared.tracing import workflow_traceable, get_trace_config

from .types import SynthesisState
from .graph import synthesis_subgraph


@workflow_traceable(name="SynthesisMapReduce", workflow_type="synthesis_mapreduce")
async def run_synthesis(
    paper_summaries: dict[str, PaperSummary],
    clusters: list[ThematicCluster],
    cluster_analyses: list[ClusterAnalysis],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    zotero_keys: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Run synthesis/writing as a standalone operation.

    Args:
        paper_summaries: DOI -> PaperSummary mapping
        clusters: List of ThematicClusters from clustering phase
        cluster_analyses: List of ClusterAnalysis from clustering phase
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        zotero_keys: Optional DOI -> Zotero key mapping

    Returns:
        Dict with final_review, references, quality_metrics, prisma_documentation
    """
    # Validate that all papers have real Zotero keys - no synthetic fallback allowed
    if zotero_keys is None:
        raise ValueError(
            "zotero_keys cannot be None - all papers must have Zotero keys from document_processing"
        )

    missing_keys = [
        doi
        for doi in paper_summaries.keys()
        if doi not in zotero_keys or not zotero_keys[doi]
    ]
    if missing_keys:
        raise ValueError(
            f"Papers missing Zotero keys: {missing_keys[:5]}"
            f"{'...' if len(missing_keys) > 5 else ''}. "
            f"Document processing may have failed for these papers."
        )

    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
    )

    initial_state = SynthesisState(
        input=input_data,
        quality_settings=quality_settings,
        paper_summaries=paper_summaries,
        clusters=clusters,
        cluster_analyses=cluster_analyses,
        zotero_keys=zotero_keys,
        introduction_draft="",
        methodology_draft="",
        thematic_section_drafts={},
        discussion_draft="",
        conclusions_draft="",
        integrated_review="",
        final_review="",
        references=[],
        citation_keys=[],
        quality_metrics=None,
        quality_passed=False,
        prisma_documentation="",
    )

    result = await synthesis_subgraph.ainvoke(initial_state, config=get_trace_config())

    return {
        "final_review": result.get("final_review", ""),
        "references": result.get("references", []),
        "citation_keys": result.get("citation_keys", []),
        "quality_metrics": result.get("quality_metrics"),
        "quality_passed": result.get("quality_passed", False),
        "prisma_documentation": result.get("prisma_documentation", ""),
        "section_drafts": {
            "introduction": result.get("introduction_draft", ""),
            "methodology": result.get("methodology_draft", ""),
            "thematic": result.get("thematic_section_drafts", {}),
            "discussion": result.get("discussion_draft", ""),
            "conclusions": result.get("conclusions_draft", ""),
        },
    }
