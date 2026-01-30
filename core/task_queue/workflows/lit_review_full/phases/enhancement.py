"""Enhancement phase (supervision + editing) execution."""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


async def run_enhancement_phase(
    lit_result: dict[str, Any],
    topic: str,
    research_questions: Optional[list[str]],
    quality: str,
    task_id: str,
    checkpoint_callback: Optional[Callable] = None,
    incremental_state: Optional[dict] = None,
) -> tuple[dict[str, Any], str]:
    """Run enhancement phase (supervision + editing).

    Args:
        lit_result: Literature review result
        topic: Research topic
        research_questions: Optional research questions
        quality: Quality level
        task_id: Task ID for incremental checkpointing
        checkpoint_callback: Optional incremental checkpoint callback
        incremental_state: Optional incremental state for resumption

    Returns:
        Tuple of (enhance_result, final_report)
    """
    from workflows.enhance import enhance_report
    from core.task_queue.incremental_state import IncrementalStateManager

    logger.info("Phase 2: Running enhancement workflow")

    # Get research questions for enhancement
    enhance_questions = research_questions or lit_result.get("research_questions", [])
    if not enhance_questions:
        enhance_questions = [f"What are the key findings regarding {topic}?"]

    # Create incremental checkpoint callback for supervision loops
    incremental_mgr = IncrementalStateManager()

    async def supervision_checkpoint(iteration: int, partial_results: dict) -> None:
        """Save incremental progress during supervision loops."""
        await incremental_mgr.save_progress(
            task_id=task_id,
            phase="supervision",
            iteration_count=iteration,
            partial_results=partial_results,
            checkpoint_interval=1,
        )

    enhance_result = await enhance_report(
        report=lit_result["final_report"],
        topic=topic,
        research_questions=enhance_questions,
        quality=quality,
        loops="all",
        paper_corpus=lit_result.get("paper_corpus"),
        paper_summaries=lit_result.get("paper_summaries"),
        zotero_keys=lit_result.get("zotero_keys"),
        run_editing=True,
        run_fact_check=False,
        checkpoint_callback=supervision_checkpoint,
        incremental_state=incremental_state,
    )

    if enhance_result.get("status") == "failed":
        logger.warning("Enhancement failed, using original lit review")
        final_report = lit_result["final_report"]
    else:
        final_report = enhance_result.get("final_report", lit_result["final_report"])

    logger.info(
        f"Enhancement complete: status={enhance_result.get('status')}, "
        f"report_length={len(final_report)}"
    )

    return enhance_result, final_report
