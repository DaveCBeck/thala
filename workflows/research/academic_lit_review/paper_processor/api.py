"""Public API for running paper processing."""

from typing import Any, Optional

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.shared.language import LanguageConfig

from .graph import paper_processing_subgraph
from .types import PaperProcessingState


async def run_paper_processing(
    papers: list[PaperMetadata],
    quality_settings: QualitySettings,
    topic: str,
    language_config: Optional[LanguageConfig] = None,
    fallback_queue: Optional[list] = None,
    paper_corpus: Optional[dict[str, PaperMetadata]] = None,
) -> dict[str, Any]:
    """Run paper processing as standalone operation.

    Args:
        papers: Papers to process
        quality_settings: Quality tier settings
        topic: Research topic
        language_config: Optional language configuration for verification.
            If provided and not English, papers will be verified to match
            the target language before extraction.
        fallback_queue: Optional pre-sorted fallback candidates for substitution
            when papers fail acquisition/processing.
        paper_corpus: Optional full paper corpus (includes overflow/near-threshold
            metadata) for FallbackManager lookups.

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
        fallback_queue=fallback_queue or [],
        fallback_substitutions=[],
        fallback_exhausted=[],
        paper_corpus=paper_corpus or {},
    )

    result = await paper_processing_subgraph.ainvoke(initial_state)
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
        # Fallback results
        "fallback_substitutions": result.get("fallback_substitutions", []),
        "fallback_exhausted": result.get("fallback_exhausted", []),
    }

    # Include language verification stats if available
    if result.get("language_verification_stats"):
        response["language_verification_stats"] = result["language_verification_stats"]
        response["language_rejected_dois"] = result.get("language_rejected_dois", [])

    return response
