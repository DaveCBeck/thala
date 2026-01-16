"""Enhancement workflows for document processing.

The main entry point is `enhance_report()` which runs:
1. Supervision loops (theoretical depth + literature expansion)
2. Editing workflow (structural coherence + flow)
"""

import logging
from typing import Any, Literal

from langsmith import traceable

from workflows.enhance.editing import editing
from workflows.enhance.supervision import (
    enhance_report as supervision_enhance,
    EnhanceInput,
    EnhanceResult,
    EnhanceState,
)

logger = logging.getLogger(__name__)

__all__ = [
    "enhance_report",
    "EnhanceInput",
    "EnhanceResult",
    "EnhanceState",
]


@traceable(run_type="chain", name="EnhanceReportFull")
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
    run_editing: bool = True,
    config: dict | None = None,
) -> dict[str, Any]:
    """Enhance an existing report with supervision loops and structural editing.

    This is the main enhancement workflow that runs:
    1. Supervision loops (if loops != "none"):
       - Loop 1: Theoretical depth (identifies and fills theoretical gaps)
       - Loop 2: Literature expansion (discovers and integrates new literature)
    2. Editing workflow (if run_editing=True):
       - Structural analysis and reorganization
       - Content generation (intros, conclusions, transitions)
       - Redundancy removal and flow polishing

    Args:
        report: Markdown report to enhance
        topic: Research topic
        research_questions: List of research questions guiding the enhancement
        quality: Quality tier affecting search depth and iteration counts.
            One of: "quick", "standard", "comprehensive", "high_quality"
        loops: Which supervision loops to run:
            - "none": Skip supervision, go straight to editing
            - "one": Only Loop 1 (theoretical depth)
            - "two": Only Loop 2 (literature expansion)
            - "all": Both loops (default)
        max_iterations_per_loop: Maximum iterations for each supervision loop (default: 3)
        paper_corpus: Optional existing paper corpus (DOI -> PaperMetadata)
        paper_summaries: Optional existing paper summaries (DOI -> PaperSummary)
        zotero_keys: Optional existing Zotero keys (DOI -> Zotero key)
        run_editing: Whether to run the editing workflow after supervision (default: True)
        config: Optional LangGraph config for tracing

    Returns:
        Dict with:
            - final_report: Final enhanced document
            - status: "success", "partial", or "failed"
            - supervision_result: Result from supervision phase (if run)
            - editing_result: Result from editing phase (if run)
            - paper_corpus: Merged paper corpus (including newly discovered)
            - paper_summaries: Merged paper summaries
            - zotero_keys: Merged Zotero keys
            - errors: List of any errors encountered

    Example:
        ```python
        from workflows.enhance import enhance_report

        result = await enhance_report(
            report=markdown_text,
            topic="Machine Learning in Healthcare",
            research_questions=["How effective is ML for diagnosis?"],
            quality="standard",
            loops="all",
            run_editing=True,
        )

        print(result["final_report"])
        print(result["status"])
        ```
    """
    errors = []
    supervision_result = None
    editing_result = None
    current_report = report

    logger.info(
        f"Starting full enhancement: topic='{topic}', quality={quality}, "
        f"loops={loops}, run_editing={run_editing}, report_length={len(report)}"
    )

    # Phase 1: Supervision loops
    if loops != "none":
        logger.info("Phase 1: Running supervision loops")
        try:
            supervision_result = await supervision_enhance(
                report=current_report,
                topic=topic,
                research_questions=research_questions,
                quality=quality,
                loops=loops,
                max_iterations_per_loop=max_iterations_per_loop,
                paper_corpus=paper_corpus,
                paper_summaries=paper_summaries,
                zotero_keys=zotero_keys,
                config=config,
            )

            current_report = supervision_result["final_report"]
            paper_corpus = supervision_result["paper_corpus"]
            paper_summaries = supervision_result["paper_summaries"]
            zotero_keys = supervision_result["zotero_keys"]

            if supervision_result.get("errors"):
                errors.extend([
                    {"phase": "supervision", **err}
                    for err in supervision_result["errors"]
                ])

            logger.info(
                f"Supervision complete: loops_run={supervision_result['loops_run']}, "
                f"report_length={len(current_report)}"
            )

        except Exception as e:
            logger.error(f"Supervision phase failed: {e}", exc_info=True)
            errors.append({"phase": "supervision", "error": str(e)})
            # Continue with original report if supervision fails
    else:
        logger.info("Phase 1: Skipping supervision loops (loops='none')")

    # Phase 2: Editing workflow
    if run_editing:
        logger.info("Phase 2: Running editing workflow")
        try:
            editing_result = await editing(
                document=current_report,
                topic=topic,
                quality=quality,
            )

            if editing_result.get("status") != "failed" and editing_result.get("final_report"):
                current_report = editing_result["final_report"]

            if editing_result.get("errors"):
                errors.extend([
                    {"phase": "editing", **err}
                    for err in editing_result["errors"]
                ])

            logger.info(
                f"Editing complete: status={editing_result.get('status')}, "
                f"report_length={len(current_report)}"
            )

        except Exception as e:
            logger.error(f"Editing phase failed: {e}", exc_info=True)
            errors.append({"phase": "editing", "error": str(e)})
    else:
        logger.info("Phase 2: Skipping editing workflow (run_editing=False)")

    # Determine overall status
    if not current_report:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"

    logger.info(f"Enhancement complete: status={status}, final_length={len(current_report)}")

    return {
        "final_report": current_report,
        "status": status,
        "supervision_result": supervision_result,
        "editing_result": editing_result,
        "paper_corpus": paper_corpus or {},
        "paper_summaries": paper_summaries or {},
        "zotero_keys": zotero_keys or {},
        "errors": errors,
    }
