"""Web-augmented literature review workflow.

Pipeline: web_scan → parallel(lit_review + web_research) → combine → enhance → evening_reads → save_and_spawn

Runs a quick web scan to sharpen research questions, then executes the existing
academic_lit_review and deep_research workflows in parallel, merges their outputs
in a dedicated combine step, and feeds the unified report into the standard
supervision → evening reads → save/spawn pipeline.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from core.task_queue.incremental_state import IncrementalStateManager

from ..base import BaseWorkflow

logger = logging.getLogger(__name__)


class LitReviewWebAugmentedWorkflow(BaseWorkflow):
    """Literature review augmented with parallel web research for recency."""

    @property
    def task_type(self) -> str:
        return "lit_review_web_augmented"

    @property
    def phases(self) -> list[str]:
        return [
            "web_scan",
            "parallel_research",
            "combine",
            "supervision",
            "editing",
            "evening_reads",
            "save_and_spawn",
            "saving",
            "complete",
        ]

    def _get_completed_phases(self, checkpoint: dict) -> set[str]:
        """Get phases completed before the checkpoint phase."""
        current_phase = checkpoint.get("phase", "")
        try:
            current_idx = self.phases.index(current_phase)
            return set(self.phases[:current_idx])
        except ValueError:
            return set()

    async def run(
        self,
        task: dict[str, Any],
        checkpoint_callback: Callable[[str], None],
        resume_from: Optional[dict] = None,
        *,
        flush_checkpoints: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """Run the web-augmented literature review workflow."""
        from .phases.web_scan import run_web_scan_phase
        from .phases.combine import run_combine_phase
        from ..lit_review_full.phases.enhancement import run_enhancement_phase
        from ..lit_review_full.phases.evening_reads import run_evening_reads_phase
        from ..lit_review_full.phases.save_and_spawn import run_save_and_spawn_phase

        topic = task["topic"]
        research_questions = task.get("research_questions")
        quality = task.get("quality", "standard")
        language = task.get("language", "en")
        date_range = task.get("date_range")
        task_id = task["id"]
        web_scan_window_days = task.get("web_scan_window_days") or 30

        # Determine which phases to skip if resuming
        completed_phases: set[str] = set()
        phase_outputs: dict[str, Any] = {}

        incremental_mgr = IncrementalStateManager()
        incremental_state = None

        if resume_from:
            completed_phases = self._get_completed_phases(resume_from)
            phase_outputs = resume_from.get("phase_outputs", {})
            current_phase = resume_from.get("phase", "")
            incremental_state = await incremental_mgr.load_progress(task_id, current_phase)
            logger.info(f"Resuming lit_review_web_augmented from {current_phase}, skipping: {completed_phases}")
        else:
            logger.info(f"Starting lit_review_web_augmented workflow: {topic[:50]}...")

        errors = []
        web_scan_result = None
        lit_result = None
        web_result = None
        combined_result = None
        enhance_result = None
        series_result = None
        final_report = None
        illustrate_task_id = None
        quartz_path = None

        # Phase 0: Web scan
        if "web_scan" in completed_phases:
            web_scan_result = phase_outputs.get("web_scan_result")
            logger.info("Skipping web_scan phase (already complete)")
        else:
            checkpoint_callback("web_scan")
            try:
                # Generate default research questions if none provided
                scan_questions = research_questions or [f"What are the key aspects of {topic}?"]

                web_scan_result = await run_web_scan_phase(
                    topic=topic,
                    research_questions=scan_questions,
                    web_scan_window_days=web_scan_window_days,
                )
                checkpoint_callback("web_scan", phase_outputs={"web_scan_result": web_scan_result})
                if flush_checkpoints:
                    await flush_checkpoints()
            except Exception as e:
                logger.error(f"Web scan failed: {e}", exc_info=True)
                errors.append({"phase": "web_scan", "error": str(e)})
                # Non-fatal: fall back to original questions
                web_scan_result = {
                    "augmented_research_questions": research_questions or [f"What are the key aspects of {topic}?"],
                    "original_research_questions": research_questions or [],
                    "recent_landscape": "",
                    "raw_results": [],
                }

        augmented_questions = web_scan_result.get("augmented_research_questions", research_questions or [])

        # Phase 1: Parallel research (lit review + web research)
        if "parallel_research" in completed_phases:
            lit_result = phase_outputs.get("lit_result")
            web_result = phase_outputs.get("web_result")
            logger.info("Skipping parallel_research phase (already complete)")
        else:
            checkpoint_callback("parallel_research")
            try:
                lit_result, web_result = await self._run_parallel_research(
                    topic=topic,
                    augmented_questions=augmented_questions,
                    quality=quality,
                    language=language,
                    date_range=date_range,
                    web_scan_window_days=web_scan_window_days,
                    phase_outputs=phase_outputs,
                    checkpoint_callback=checkpoint_callback,
                )
                checkpoint_callback(
                    "parallel_research",
                    phase_outputs={
                        "web_scan_result": web_scan_result,
                        "lit_result": lit_result,
                        "web_result": web_result,
                    },
                )
                if flush_checkpoints:
                    await flush_checkpoints()
            except Exception as e:
                logger.error(f"Parallel research failed: {e}", exc_info=True)
                errors.append({"phase": "parallel_research", "error": str(e)})
                return {"status": "failed", "errors": errors}

        # Validate we have at least one result
        if not lit_result and not web_result:
            logger.error("Both research legs failed")
            return {
                "status": "failed",
                "errors": errors + [{"phase": "parallel_research", "error": "both legs failed"}],
            }

        # Phase 2: Combine
        if "combine" in completed_phases:
            combined_result = phase_outputs.get("combined_result")
            logger.info("Skipping combine phase (already complete)")
        else:
            checkpoint_callback("combine")
            try:
                combined_result = await run_combine_phase(
                    lit_result=lit_result
                    or {"final_report": "", "paper_corpus": {}, "paper_summaries": {}, "zotero_keys": {}},
                    web_result=web_result or {"final_report": ""},
                    topic=topic,
                    augmented_research_questions=augmented_questions,
                    recent_landscape=web_scan_result.get("recent_landscape", ""),
                )
                checkpoint_callback(
                    "combine",
                    phase_outputs={
                        "web_scan_result": web_scan_result,
                        "lit_result": lit_result,
                        "web_result": web_result,
                        "combined_result": combined_result,
                    },
                )
                if flush_checkpoints:
                    await flush_checkpoints()
            except Exception as e:
                logger.error(f"Combine phase failed: {e}", exc_info=True)
                errors.append({"phase": "combine", "error": str(e)})
                # Fall back to academic report if available
                if lit_result and lit_result.get("final_report"):
                    combined_result = lit_result
                else:
                    return {"status": "failed", "errors": errors}

        # Phase 3: Enhancement (supervision + editing) — reused from lit_review_full
        if "supervision" in completed_phases and "editing" in completed_phases:
            enhance_result = phase_outputs.get("enhance_result")
            final_report = phase_outputs.get("final_report") or (
                enhance_result.get("final_report") if enhance_result else None
            )
            logger.info("Skipping supervision/editing phases (already complete)")
        else:
            checkpoint_callback("supervision")
            try:
                supervision_incremental_state = (
                    incremental_state if incremental_state and incremental_state.get("phase") == "supervision" else None
                )

                enhance_result, final_report = await run_enhancement_phase(
                    lit_result=combined_result,
                    topic=topic,
                    research_questions=augmented_questions,
                    quality=quality,
                    task_id=task_id,
                    incremental_state=supervision_incremental_state,
                )

                if final_report and enhance_result:
                    from ..lit_review_full.phases.methodology import append_editorial_summary

                    final_report = append_editorial_summary(final_report, enhance_result)

                if enhance_result.get("errors"):
                    errors.extend(enhance_result["errors"])

                checkpoint_callback(
                    "editing",
                    phase_outputs={
                        "web_scan_result": web_scan_result,
                        "lit_result": lit_result,
                        "web_result": web_result,
                        "combined_result": combined_result,
                        "enhance_result": enhance_result,
                        "final_report": final_report,
                    },
                )
                if flush_checkpoints:
                    await flush_checkpoints()
                await incremental_mgr.clear_progress(task_id)

            except Exception as e:
                logger.error(f"Enhancement failed: {e}", exc_info=True)
                errors.append({"phase": "enhancement", "error": str(e)})
                if combined_result and combined_result.get("final_report"):
                    final_report = combined_result["final_report"]

        # Phase 4: Evening reads — reused from lit_review_full
        if "evening_reads" in completed_phases:
            series_result = phase_outputs.get("series_result")
            logger.info("Skipping evening_reads phase (already complete)")
        else:
            checkpoint_callback("evening_reads")
            try:
                series_result = await run_evening_reads_phase(
                    final_report=final_report,
                    category=task.get("category", ""),
                )
                checkpoint_callback(
                    "evening_reads",
                    phase_outputs={
                        "web_scan_result": web_scan_result,
                        "lit_result": lit_result,
                        "web_result": web_result,
                        "combined_result": combined_result,
                        "enhance_result": enhance_result,
                        "final_report": final_report,
                        "series_result": series_result,
                    },
                )
                if flush_checkpoints:
                    await flush_checkpoints()
            except Exception as e:
                logger.error(f"Evening reads failed: {e}", exc_info=True)
                errors.append({"phase": "evening_reads", "error": str(e)})
                series_result = {"final_outputs": []}

        # Phase 5: Save and spawn — reused from lit_review_full
        if "save_and_spawn" in completed_phases:
            illustrate_task_id = phase_outputs.get("illustrate_task_id")
            quartz_path = phase_outputs.get("quartz_path")
            logger.info("Skipping save_and_spawn phase (already complete)")
        else:
            checkpoint_callback("save_and_spawn")
            final_outputs = series_result.get("final_outputs", []) if series_result else []
            if final_report and final_outputs:
                try:
                    spawn_result = await run_save_and_spawn_phase(
                        task=task,
                        final_report=final_report,
                        final_outputs=final_outputs,
                        get_output_dir_fn=self.get_output_dir,
                        slugify_fn=self.slugify,
                    )
                    illustrate_task_id = spawn_result.get("illustrate_task_id")
                    quartz_path = spawn_result.get("quartz_path")
                    checkpoint_callback(
                        "save_and_spawn",
                        phase_outputs={
                            "web_scan_result": web_scan_result,
                            "lit_result": lit_result,
                            "web_result": web_result,
                            "combined_result": combined_result,
                            "enhance_result": enhance_result,
                            "final_report": final_report,
                            "series_result": series_result,
                            "illustrate_task_id": illustrate_task_id,
                            "quartz_path": quartz_path,
                        },
                    )
                    if flush_checkpoints:
                        await flush_checkpoints()
                except Exception as e:
                    logger.error(f"Save and spawn failed: {e}", exc_info=True)
                    errors.append({"phase": "save_and_spawn", "error": str(e)})

        # Determine status
        if not combined_result or not combined_result.get("final_report"):
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return {
            "status": status,
            "topic": topic,
            "lit_review": lit_result,
            "web_research": web_result,
            "combined": combined_result,
            "enhance": enhance_result,
            "series": series_result,
            "final_report": final_report,
            "illustrate_task_id": illustrate_task_id,
            "quartz_path": quartz_path,
            "errors": errors,
        }

    async def _run_parallel_research(
        self,
        topic: str,
        augmented_questions: list[str],
        quality: str,
        language: str,
        date_range: Optional[tuple],
        web_scan_window_days: int,
        phase_outputs: dict[str, Any],
        checkpoint_callback: Callable,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """Run lit review and web research in parallel.

        Each leg checkpoints its result immediately on completion so that
        if one leg fails, the other's result is preserved for resume.
        """
        from datetime import datetime, timedelta

        from workflows.research.web_research import deep_research
        from workflows.research.web_research.state import RecencyFilter

        from ..lit_review_full.phases.lit_review import run_lit_review_phase

        # Check if either leg already completed (partial resume)
        existing_lit = phase_outputs.get("lit_result")
        existing_web = phase_outputs.get("web_result")

        # Build recency filter for web research
        after_date = (datetime.now() - timedelta(days=web_scan_window_days)).strftime("%Y-%m-%d")
        recency_filter = RecencyFilter(after_date=after_date, quota=0.3)

        # Format questions as a query string for deep_research
        questions_query = f"{topic}\n\nResearch questions:\n" + "\n".join(f"- {q}" for q in augmented_questions)

        async def _run_lit_leg() -> Optional[dict]:
            if existing_lit:
                logger.info("Lit review leg: using cached result from checkpoint")
                return existing_lit
            try:
                result = await run_lit_review_phase(
                    topic=topic,
                    research_questions=augmented_questions,
                    quality=quality,
                    language=language,
                    date_range=date_range,
                )
                # Checkpoint immediately so result survives if the other leg fails
                checkpoint_callback(
                    "parallel_research",
                    phase_outputs={**phase_outputs, "lit_result": result},
                )
                return result
            except Exception as e:
                logger.error(f"Lit review leg failed: {e}", exc_info=True)
                return None

        async def _run_web_leg() -> Optional[dict]:
            if existing_web:
                logger.info("Web research leg: using cached result from checkpoint")
                return existing_web
            try:
                result = await deep_research(
                    query=questions_query,
                    quality=quality,
                    language=language,
                    recency_filter=recency_filter,
                )
                checkpoint_callback(
                    "parallel_research",
                    phase_outputs={**phase_outputs, "web_result": result},
                )
                return result
            except Exception as e:
                logger.error(f"Web research leg failed: {e}", exc_info=True)
                return None

        # Run both legs concurrently; return_exceptions=True so one failure doesn't kill the other
        lit_result, web_result = await asyncio.gather(
            _run_lit_leg(),
            _run_web_leg(),
            return_exceptions=True,
        )

        # Handle exceptions from gather
        if isinstance(lit_result, Exception):
            logger.error(f"Lit review leg raised: {lit_result}")
            lit_result = None
        if isinstance(web_result, Exception):
            logger.error(f"Web research leg raised: {web_result}")
            web_result = None

        logger.info(
            f"Parallel research complete: lit={'OK' if lit_result else 'FAILED'}, "
            f"web={'OK' if web_result else 'FAILED'}"
        )

        return lit_result, web_result

    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save outputs — delegates to lit_review_full's persistence."""
        from ..lit_review_full.output.persistence import save_workflow_outputs

        return save_workflow_outputs(
            task=task,
            result=result,
            get_output_dir_fn=self.get_output_dir,
            generate_timestamp_fn=self.generate_timestamp,
            slugify_fn=self.slugify,
        )
