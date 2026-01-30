"""
Workflow execution with checkpoint and budget integration.

Provides:
- Execute workflows with checkpoint callbacks
- Budget-aware execution
- Resume from checkpoints
- LangSmith project isolation
"""

import asyncio
import logging
import os
import uuid
from typing import Optional

from core.logging import end_run, start_run

from .budget_tracker import BudgetTracker
from .checkpoint import CheckpointManager
from .queue_manager import TaskQueueManager
from .schemas import Task, WorkflowCheckpoint
from .shutdown import ShutdownCoordinator
from .workflows import get_workflow, DEFAULT_WORKFLOW_TYPE

logger = logging.getLogger(__name__)

# Queue uses a dedicated LangSmith project for budget isolation
QUEUE_PROJECT = os.getenv("THALA_QUEUE_PROJECT", "thala-queue")


async def run_task_workflow(
    task: Task,
    queue_manager: TaskQueueManager,
    checkpoint_mgr: CheckpointManager,
    budget_tracker: BudgetTracker,
    resume_from: Optional[WorkflowCheckpoint] = None,
    shutdown_coordinator: Optional[ShutdownCoordinator] = None,
) -> dict:
    """Run a workflow for any task type via registry dispatch.

    Dispatches to the appropriate workflow based on task_type field.
    Handles checkpoint updates, budget checks, and status management.

    Args:
        task: Task to process (TopicTask, WebResearchTask, etc.)
        queue_manager: Queue manager instance
        checkpoint_mgr: Checkpoint manager instance
        budget_tracker: Budget tracker instance
        resume_from: Optional checkpoint to resume from
        shutdown_coordinator: Optional coordinator for graceful shutdown

    Returns:
        Workflow result dict with status, outputs, etc.
    """
    # Set LangSmith project for queue runs (isolates budget from manual testing)
    os.environ["LANGSMITH_PROJECT"] = QUEUE_PROJECT
    logger.info(f"Using LangSmith project: {QUEUE_PROJECT}")

    # Get task type and workflow
    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
    workflow = get_workflow(task_type)

    task_id = task["id"]
    task_identifier = workflow.get_task_identifier(task)

    # Start logging run (triggers log rotation on first write to each module)
    start_run(task_id)

    try:
        # Generate langsmith_run_id (or use existing if resuming)
        if resume_from:
            langsmith_run_id = resume_from["langsmith_run_id"]
            logger.info(f"Resuming task {task_id[:8]} from phase {resume_from['phase']}")
        else:
            langsmith_run_id = str(uuid.uuid4())
            logger.info(f"Starting task {task_id[:8]} ({task_type}): {task_identifier}")

        # Mark as started
        queue_manager.mark_started(task_id, langsmith_run_id)
        await checkpoint_mgr.start_work(task_id, task_type, langsmith_run_id)

        # Track pending checkpoint tasks to avoid race conditions with cleanup
        pending_checkpoint_tasks: list[asyncio.Task] = []

        async def await_pending_checkpoints():
            """Await all pending checkpoint tasks before cleanup/completion."""
            if pending_checkpoint_tasks:
                await asyncio.gather(*pending_checkpoint_tasks, return_exceptions=True)
                pending_checkpoint_tasks.clear()

        # Create checkpoint callback (sync wrapper for async operations)
        def checkpoint_callback(phase: str, phase_outputs: dict | None = None, **kwargs) -> None:
            """Update checkpoint during workflow execution.

            This is a sync function that schedules async work, matching the
            Callable[[str], None] interface expected by workflows.
            """
            async def _update():
                await checkpoint_mgr.update_checkpoint(task_id, phase, phase_outputs=phase_outputs, **kwargs)

            # Schedule the async checkpoint update and track it
            task = asyncio.create_task(_update())
            pending_checkpoint_tasks.append(task)

            # Sync operations
            queue_manager.update_phase(task_id, phase)

            # Check budget between phases
            should_proceed, reason = budget_tracker.should_proceed()
            if not should_proceed:
                logger.warning(f"Budget limit reached during {phase}: {reason}")
                # Don't raise - let the current phase complete

            # Check for shutdown request
            if shutdown_coordinator and shutdown_coordinator.shutdown_requested:
                logger.info(f"Shutdown requested during phase {phase}")
                # Don't raise - let the current checkpoint complete

        # Run the workflow (pass shutdown_coordinator if workflow supports it)
        result = await workflow.run(task, checkpoint_callback, resume_from)

        # Save outputs
        checkpoint_callback("saving")
        output_paths = workflow.save_outputs(task, result)
        result["output_paths"] = output_paths

        if output_paths:
            logger.info(f"Saved outputs: {list(output_paths.keys())}")

        # Ensure all pending checkpoint writes complete before marking done
        await await_pending_checkpoints()

        # Mark complete or failed
        if result.get("status") == "success":
            queue_manager.mark_completed(task_id)
            await checkpoint_mgr.complete_work(task_id)
            logger.info(f"Task {task_id[:8]} completed successfully")
        elif result.get("status") == "partial":
            # Partial success - mark complete but log warnings
            queue_manager.mark_completed(task_id)
            await checkpoint_mgr.complete_work(task_id)
            logger.warning(f"Task {task_id[:8]} completed with errors: {result.get('errors')}")
        else:
            error = str(result.get("errors", "Unknown error"))
            queue_manager.mark_failed(task_id, error)
            await checkpoint_mgr.fail_work(task_id)
            logger.error(f"Task {task_id[:8]} failed: {error}")

        return result

    except asyncio.CancelledError:
        logger.info(f"Task {task_id[:8]} cancelled - checkpoint preserved for resumption")
        # Don't mark as failed - preserves checkpoint for later resumption
        raise

    except Exception as e:
        logger.error(f"Task {task_id[:8]} failed with exception: {e}")
        # Ensure pending checkpoints complete before marking failed
        await await_pending_checkpoints()
        queue_manager.mark_failed(task_id, str(e))
        await checkpoint_mgr.fail_work(task_id)
        raise

    finally:
        # End logging run (best-effort cleanup)
        end_run()
