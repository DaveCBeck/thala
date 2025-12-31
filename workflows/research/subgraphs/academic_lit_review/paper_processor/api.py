"""Public API for running paper processing."""

from typing import Any

from workflows.research.subgraphs.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)

from .graph import paper_processing_subgraph
from .types import PaperProcessingState


async def run_paper_processing(
    papers: list[PaperMetadata],
    quality_settings: QualitySettings,
    topic: str,
) -> dict[str, Any]:
    """Run paper processing as standalone operation.

    Args:
        papers: Papers to process
        quality_settings: Quality tier settings
        topic: Research topic

    Returns:
        Dict with paper_summaries, elasticsearch_ids, zotero_keys
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=[],
        quality="standard",
        date_range=None,
        include_books=False,
        focus_areas=None,
        exclude_terms=None,
        max_papers=None,
    )

    initial_state = PaperProcessingState(
        input=input_data,
        quality_settings=quality_settings,
        papers_to_process=papers,
        acquired_papers={},
        acquisition_failed=[],
        processing_results={},
        processing_failed=[],
        paper_summaries={},
        elasticsearch_ids={},
        zotero_keys={},
    )

    result = await paper_processing_subgraph.ainvoke(initial_state)
    return {
        "paper_summaries": result.get("paper_summaries", {}),
        "elasticsearch_ids": result.get("elasticsearch_ids", {}),
        "zotero_keys": result.get("zotero_keys", {}),
        "acquisition_failed": result.get("acquisition_failed", []),
        "processing_failed": result.get("processing_failed", []),
    }
