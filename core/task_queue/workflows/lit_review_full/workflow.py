"""Full literature review workflow.

Pipeline: lit_review → enhance (supervision+editing) → evening_reads → save_and_spawn

This is the complete academic research workflow that:
1. Discovers and processes academic papers
2. Generates an initial literature review
3. Enhances with supervision loops and editing
4. Produces an evening reads article series
5. Saves unillustrated articles and spawns illustrate_and_export task
"""

import logging
from typing import Any, Awaitable, Callable, Optional

from core.task_queue.incremental_state import IncrementalStateManager

from ..base import BaseWorkflow

logger = logging.getLogger(__name__)


class LitReviewFullWorkflow(BaseWorkflow):
    """Full literature review with enhancement and article series generation."""

    @property
    def task_type(self) -> str:
        return "lit_review_full"

    @property
    def phases(self) -> list[str]:
        return [
            "lit_review",
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
        """Run the full literature review workflow.

        Args:
            task: LitReviewTask/TopicTask with topic, research_questions, quality, etc.
            checkpoint_callback: Progress callback
            resume_from: Optional checkpoint for resumption
            flush_checkpoints: Optional async function to await pending checkpoint writes

        Returns:
            Dict with status, lit_review, enhance, series, illustrated results
        """
        from .phases.lit_review import run_lit_review_phase
        from .phases.enhancement import run_enhancement_phase
        from .phases.evening_reads import run_evening_reads_phase
        from .phases.save_and_spawn import run_save_and_spawn_phase

        topic = task["topic"]
        research_questions = task.get("research_questions")
        quality = task.get("quality", "standard")
        language = task.get("language", "en")
        date_range = task.get("date_range")
        task_id = task["id"]

        # Determine which phases to skip if resuming
        completed_phases: set[str] = set()
        phase_outputs: dict[str, Any] = {}

        # Create incremental state manager for mid-phase checkpointing
        incremental_mgr = IncrementalStateManager()

        # Load incremental state if resuming (for mid-phase progress)
        incremental_state = None

        if resume_from:
            completed_phases = self._get_completed_phases(resume_from)
            phase_outputs = resume_from.get("phase_outputs", {})
            current_phase = resume_from.get("phase", "")

            # Load incremental state for the current phase (mid-phase progress)
            incremental_state = await incremental_mgr.load_progress(task_id, current_phase)
            if incremental_state:
                logger.info(
                    f"Loaded incremental state for phase '{current_phase}': "
                    f"iteration {incremental_state.get('iteration_count', 0)}"
                )

            logger.info(f"Resuming lit_review_full from {current_phase}, skipping completed phases: {completed_phases}")
        else:
            logger.info(f"Starting lit_review_full workflow: {topic[:50]}...")

        errors = []
        lit_result = None
        enhance_result = None
        series_result = None
        final_report = None
        illustrate_task_id = None
        quartz_path = None

        # Phase 1: Literature review
        if "lit_review" in completed_phases:
            lit_result = phase_outputs.get("lit_result")
            logger.info("Skipping lit_review phase (already complete)")
        else:
            checkpoint_callback("lit_review")
            try:
                lit_result = await run_lit_review_phase(
                    topic=topic,
                    research_questions=research_questions,
                    quality=quality,
                    language=language,
                    date_range=date_range,
                    category=task.get("category"),
                )
                checkpoint_callback("lit_review", phase_outputs={"lit_result": lit_result})
                # Flush to ensure phase_outputs are persisted before next phase
                if flush_checkpoints:
                    await flush_checkpoints()
            except Exception as e:
                logger.error(f"Literature review failed: {e}", exc_info=True)
                errors.append({"phase": "lit_review", "error": str(e)})
                return {"status": "failed", "errors": errors}

        # Validate lit_result exists (could be missing from corrupted checkpoint)
        if lit_result is None:
            logger.error("lit_result is None - checkpoint may be corrupted")
            errors.append({"phase": "lit_review", "error": "lit_result missing from checkpoint"})
            return {"status": "failed", "errors": errors}

        # Phase 2: Enhancement (supervision + editing)
        if "supervision" in completed_phases and "editing" in completed_phases:
            enhance_result = phase_outputs.get("enhance_result")
            final_report = phase_outputs.get("final_report") or (
                enhance_result.get("final_report") if enhance_result else None
            )
            logger.info("Skipping supervision/editing phases (already complete)")
        else:
            checkpoint_callback("supervision")
            try:
                # Pass incremental_state only if we're resuming the supervision phase
                supervision_incremental_state = (
                    incremental_state if incremental_state and incremental_state.get("phase") == "supervision" else None
                )

                enhance_result, final_report = await run_enhancement_phase(
                    lit_result=lit_result,
                    topic=topic,
                    research_questions=research_questions,
                    quality=quality,
                    task_id=task_id,
                    incremental_state=supervision_incremental_state,
                )

                # Append editorial transparency to methodology section
                if final_report and enhance_result:
                    from .phases.methodology import append_editorial_summary

                    final_report = append_editorial_summary(final_report, enhance_result)

                if enhance_result.get("errors"):
                    errors.extend(enhance_result["errors"])

                # Save phase outputs FIRST before clearing incremental state
                checkpoint_callback(
                    "editing",
                    phase_outputs={
                        "lit_result": lit_result,
                        "enhance_result": enhance_result,
                        "final_report": final_report,
                    },
                )

                # Ensure checkpoint is persisted before clearing incremental state
                if flush_checkpoints:
                    await flush_checkpoints()

                # THEN clear incremental state
                await incremental_mgr.clear_progress(task_id)

            except Exception as e:
                logger.error(f"Enhancement failed: {e}", exc_info=True)
                errors.append({"phase": "enhancement", "error": str(e)})
                # Fall back to lit_result's report if enhancement failed
                if lit_result and lit_result.get("final_report"):
                    final_report = lit_result["final_report"]

        # Phase 3: Evening reads article series
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
                        "lit_result": lit_result,
                        "enhance_result": enhance_result,
                        "final_report": final_report,
                        "series_result": series_result,
                    },
                )
                # Flush to ensure phase_outputs are persisted before next phase
                if flush_checkpoints:
                    await flush_checkpoints()
            except Exception as e:
                logger.error(f"Evening reads failed: {e}", exc_info=True)
                errors.append({"phase": "evening_reads", "error": str(e)})
                series_result = {"final_outputs": []}

        # Phase 4: Save articles to disk, export lit review to Quartz, spawn illustrate_and_export task
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
                            "lit_result": lit_result,
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
        if not lit_result or not lit_result.get("final_report"):
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "success"

        return {
            "status": status,
            "topic": topic,
            "lit_review": lit_result,
            "enhance": enhance_result,
            "series": series_result,
            "final_report": final_report,
            "illustrate_task_id": illustrate_task_id,
            "quartz_path": quartz_path,
            "errors": errors,
        }

    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save literature review and article series to .thala/output/ directory.

        Note: Illustrated versions are already saved during the illustrate phase.
        This method saves additional metadata and non-illustrated versions.

        Args:
            task: Task data
            result: Workflow result

        Returns:
            Dict with paths to saved files
        """
        from .output.persistence import save_workflow_outputs

        return save_workflow_outputs(
            task=task,
            result=result,
            get_output_dir_fn=self.get_output_dir,
            generate_timestamp_fn=self.generate_timestamp,
            slugify_fn=self.slugify,
        )
