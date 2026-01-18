"""Public API for running paper processing."""

from typing import Any, Optional

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.shared.language import LanguageConfig
from workflows.shared.tracing import workflow_traceable, get_trace_config

from .graph import paper_processing_subgraph
from .types import PaperProcessingState


@workflow_traceable(name="PaperProcessor", workflow_type="paper_processor")
async def run_paper_processing(
    papers: list[PaperMetadata],
    quality_settings: QualitySettings,
    topic: str,
    language_config: Optional[LanguageConfig] = None,
) -> dict[str, Any]:
    """Run paper processing as standalone operation.

    Args:
        papers: Papers to process
        quality_settings: Quality tier settings
        topic: Research topic
        language_config: Optional language configuration for verification.
            If provided and not English, papers will be verified to match
            the target language before extraction.

    Returns:
        Dict with paper_summaries, elasticsearch_ids, zotero_keys,
        and language_verification_stats if language verification was performed.
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=[],
        quality="standard",
        date_range=None,
    )

    initial_state = PaperProcessingState(
        input=input_data,
        quality_settings=quality_settings,
        papers_to_process=papers,
        language_config=language_config,
        acquired_papers={},
        acquisition_failed=[],
        processing_results={},
        processing_failed=[],
        language_rejected_dois=[],
        paper_summaries={},
        elasticsearch_ids={},
        zotero_keys={},
    )

    result = await paper_processing_subgraph.ainvoke(
        initial_state, config=get_trace_config()
    )
    paper_summaries = result.get("paper_summaries", {})

    response = {
        "paper_summaries": paper_summaries,
        "elasticsearch_ids": result.get("elasticsearch_ids", {}),
        "zotero_keys": result.get("zotero_keys", {}),
        "processed_dois": list(paper_summaries.keys()),
        "failed_dois": result.get("processing_failed", []),
        # Legacy keys for backwards compatibility
        "acquisition_failed": result.get("acquisition_failed", []),
        "processing_failed": result.get("processing_failed", []),
    }

    # Include language verification stats if available
    if result.get("language_verification_stats"):
        response["language_verification_stats"] = result["language_verification_stats"]
        response["language_rejected_dois"] = result.get("language_rejected_dois", [])

    return response
