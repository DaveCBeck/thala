"""Loop 2 utility functions."""

from workflows.academic_lit_review.state import LitReviewInput
from workflows.academic_lit_review.quality_presets import QualitySettings

from .graph import Loop2State, loop2_graph


async def run_loop2_standalone(
    review: str,
    paper_corpus: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    input_data: LitReviewInput,
    quality_settings: QualitySettings,
    max_iterations: int = 3,
    config: dict | None = None,
) -> dict:
    """Run Loop 2 as standalone operation for testing.

    Args:
        review: Current review text
        paper_corpus: DOI -> PaperMetadata mapping
        paper_summaries: DOI -> PaperSummary mapping
        zotero_keys: DOI -> Zotero key mapping
        input_data: LitReviewInput with topic and research questions
        quality_settings: Quality tier settings
        max_iterations: Maximum expansion iterations (default: 3)
        config: Optional LangGraph config with run_id and run_name for tracing

    Returns:
        Dict with:
            - current_review: Updated review text
            - paper_summaries: Merged paper summaries
            - zotero_keys: Merged Zotero keys
            - paper_corpus: Merged corpus
            - explored_bases: List of literature bases explored
            - iteration: Final iteration count
            - is_complete: Whether loop completed
    """
    initial_state = Loop2State(
        current_review=review,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
        input=input_data,
        quality_settings=quality_settings,
        iteration=0,
        max_iterations=max_iterations,
        explored_bases=[],
        is_complete=False,
        decision=None,
        errors=[],
        iterations_failed=0,
        consecutive_failures=0,
        integration_failed=False,
        mini_review_failed=False,
    )

    if config:
        result = await loop2_graph.ainvoke(initial_state, config=config)
    else:
        result = await loop2_graph.ainvoke(initial_state)

    # Calculate newly added zotero keys (not in original input)
    final_zotero_keys = result.get("zotero_keys", zotero_keys)
    added_zotero_keys = {
        doi: key for doi, key in final_zotero_keys.items()
        if doi not in zotero_keys
    }

    return {
        "current_review": result.get("current_review", review),
        "paper_summaries": result.get("paper_summaries", paper_summaries),
        "zotero_keys": result.get("zotero_keys", zotero_keys),
        "added_zotero_keys": added_zotero_keys,
        "paper_corpus": result.get("paper_corpus", paper_corpus),
        "explored_bases": result.get("explored_bases", []),
        "iteration": result.get("iteration", 1),
        "is_complete": result.get("is_complete", False),
        "errors": result.get("errors", []),
        "iterations_failed": result.get("iterations_failed", 0),
    }
