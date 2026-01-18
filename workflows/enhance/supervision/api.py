"""API entry point for report enhancement workflow.

Provides `enhance_report()` - the main function to enhance an existing
markdown report with theoretical depth (Loop 1) and literature expansion (Loop 2).
"""

import logging
from typing import Any, Literal

from langsmith import traceable

from workflows.enhance.supervision.builder import create_enhancement_graph
from workflows.enhance.supervision.types import EnhanceInput, EnhanceResult, EnhanceState
from workflows.research.academic_lit_review.quality_presets import QUALITY_PRESETS

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EnhanceReport")
async def enhance_report(
    report: str,
    topic: str,
    research_questions: list[str],
    quality: Literal["quick", "standard", "comprehensive", "high_quality"] = "standard",
    loops: Literal["none", "one", "two", "all"] = "all",
    max_iterations_per_loop: int = 3,
    paper_corpus: dict[str, Any] | None = None,
    paper_summaries: dict[str, Any] | None = None,
    zotero_keys: dict[str, str] | None = None,
    config: dict | None = None,
) -> EnhanceResult:
    """Enhance an existing report with theoretical depth and literature expansion.

    This workflow accepts a markdown report and runs:
    - Loop 1 (theoretical depth): Identifies and fills theoretical gaps
    - Loop 2 (literature expansion): Discovers and integrates new literature bases

    Args:
        report: Markdown report to enhance
        topic: Research topic
        research_questions: List of research questions guiding the enhancement
        quality: Quality tier affecting search depth and iteration counts.
            One of: "quick", "standard", "comprehensive", "high_quality"
        loops: Which loops to run:
            - "none": No loops, returns input unchanged
            - "one": Only Loop 1 (theoretical depth)
            - "two": Only Loop 2 (literature expansion)
            - "all": Both loops (default)
        max_iterations_per_loop: Maximum iterations for each loop (default: 3)
        paper_corpus: Optional existing paper corpus (DOI -> PaperMetadata)
        paper_summaries: Optional existing paper summaries (DOI -> PaperSummary)
        zotero_keys: Optional existing Zotero keys (DOI -> Zotero key)
        config: Optional LangGraph config for tracing

    Returns:
        EnhanceResult with:
            - final_report: Enhanced markdown report
            - review_loop1: Report after Loop 1 (if run)
            - review_loop2: Report after Loop 2 (if run)
            - loops_run: List of loops that were executed
            - paper_corpus: Merged paper corpus (including newly discovered)
            - paper_summaries: Merged paper summaries
            - zotero_keys: Merged Zotero keys
            - completion_reason: Summary of how enhancement completed
            - errors: List of any errors encountered

    Example:
        ```python
        result = await enhance_report(
            report=sample_markdown,
            topic="Test Topic",
            research_questions=["How does X work?"],
            quality="quick",
            loops="all",
        )
        print(result["final_report"])
        ```
    """
    # Get quality settings
    quality_settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])

    # Build input
    enhance_input: EnhanceInput = {
        "report": report,
        "topic": topic,
        "research_questions": research_questions,
        "quality": quality,
    }

    # Build initial state
    initial_state: EnhanceState = {
        "current_review": report,
        "final_review": None,
        "review_loop1": None,
        "review_loop2": None,
        "paper_corpus": paper_corpus or {},
        "paper_summaries": paper_summaries or {},
        "zotero_keys": zotero_keys or {},
        "input": enhance_input,
        "quality_settings": quality_settings,
        "max_iterations_per_loop": max_iterations_per_loop,
        "loop_progress": [],
        "loop1_result": None,
        "loop2_result": None,
        "completion_reason": "",
        "is_complete": False,
        "errors": [],
    }

    logger.info(
        f"Starting enhancement: topic='{topic}', quality={quality}, loops={loops}, "
        f"report_length={len(report)}, existing_papers={len(paper_corpus or {})}"
    )

    # Create and run the graph
    graph = create_enhancement_graph(loops=loops)

    if config:
        final_state = await graph.ainvoke(initial_state, config=config)
    else:
        final_state = await graph.ainvoke(initial_state)

    # Extract loops that were run from progress
    loops_run = [
        entry.get("loop", "unknown")
        for entry in final_state.get("loop_progress", [])
    ]

    logger.info(
        f"Enhancement complete: loops_run={loops_run}, "
        f"final_length={len(final_state.get('final_review', ''))}"
    )

    return EnhanceResult(
        final_report=final_state.get("final_review", report),
        review_loop1=final_state.get("review_loop1"),
        review_loop2=final_state.get("review_loop2"),
        loops_run=loops_run,
        paper_corpus=final_state.get("paper_corpus", {}),
        paper_summaries=final_state.get("paper_summaries", {}),
        zotero_keys=final_state.get("zotero_keys", {}),
        completion_reason=final_state.get("completion_reason", ""),
        errors=final_state.get("errors", []),
    )
