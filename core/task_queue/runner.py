"""
Workflow runner with checkpoint and budget integration.

Provides:
- Run tasks from queue with checkpoint callbacks
- Registry-based dispatch to workflow implementations
- Resume incomplete work
- Budget-aware execution with adaptive delays
- Continuous queue processing loop
- Graceful shutdown via SIGINT/SIGTERM

Uses THALA_QUEUE_PROJECT for LangSmith tracing to isolate queue costs
from manual testing/development costs.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from core.logging import end_run, start_run

from .budget_tracker import BudgetTracker
from .checkpoint_manager import CheckpointManager
from .incremental_state import IncrementalStateManager
from .pricing import format_cost
from .queue_manager import TaskQueueManager
from .schemas import Task, WorkflowCheckpoint
from .shutdown import ShutdownCoordinator, get_shutdown_coordinator
from .workflows import get_workflow, DEFAULT_WORKFLOW_TYPE

logger = logging.getLogger(__name__)

# Queue uses a dedicated LangSmith project for budget isolation
QUEUE_PROJECT = os.getenv("THALA_QUEUE_PROJECT", "thala-queue")


def _find_bypass_task(queue_manager: TaskQueueManager) -> Optional[Task]:
    """Find a pending task that bypasses concurrency limits.

    Bypass tasks (like publish_series) can run regardless of stagger_hours
    or max_concurrent settings.

    Args:
        queue_manager: Queue manager instance

    Returns:
        First eligible bypass task, or None
    """
    from .schemas import TaskStatus

    pending_tasks = queue_manager.list_tasks(status=TaskStatus.PENDING)

    for task in pending_tasks:
        task_type = task.get("task_type", "lit_review_full")
        workflow = get_workflow(task_type)
        if getattr(workflow, "bypass_concurrency", False):
            return task

    return None


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

        # Create checkpoint callback
        async def checkpoint_callback(phase: str, phase_outputs: dict | None = None, **kwargs) -> None:
            """Update checkpoint during workflow execution."""
            await checkpoint_mgr.update_checkpoint(task_id, phase, phase_outputs=phase_outputs, **kwargs)
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
        await checkpoint_callback("saving")
        output_paths = workflow.save_outputs(task, result)
        result["output_paths"] = output_paths

        if output_paths:
            logger.info(f"Saved outputs: {list(output_paths.keys())}")

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
        queue_manager.mark_failed(task_id, str(e))
        await checkpoint_mgr.fail_work(task_id)
        raise

    finally:
        # End logging run (best-effort cleanup)
        end_run()


async def run_single_task(
    queue_dir: Optional[Path] = None,
    skip_resume: bool = False,
    dry_run: bool = False,
    shutdown_coordinator: Optional[ShutdownCoordinator] = None,
) -> Optional[dict]:
    """Run the next eligible task from the queue.

    This is a convenience wrapper around run_queue_loop(max_tasks=1).

    Args:
        queue_dir: Override queue directory (for testing)
        skip_resume: Skip incomplete work and start with a new task
        dry_run: If True, only show what would run
        shutdown_coordinator: Optional coordinator for graceful shutdown

    Returns:
        Workflow result dict, or None if nothing to run
    """
    # Delegate to run_queue_loop with max_tasks=1
    return await run_queue_loop(
        queue_dir=queue_dir,
        max_tasks=1,
        skip_resume=skip_resume,
        dry_run=dry_run,
        install_signal_handlers=False,  # Don't install handlers for single task
        shutdown_coordinator=shutdown_coordinator,
    )


async def run_queue_loop(
    queue_dir: Optional[Path] = None,
    max_tasks: Optional[int] = None,
    skip_resume: bool = False,
    dry_run: bool = False,
    check_interval: float = 300.0,  # 5 minutes
    install_signal_handlers: bool = True,
    shutdown_coordinator: Optional[ShutdownCoordinator] = None,
) -> Optional[dict]:
    """Continuously process tasks from queue with graceful shutdown support.

    Optionally installs signal handlers for SIGINT/SIGTERM to enable graceful shutdown.
    When a shutdown signal is received, the current checkpoint is saved
    and the loop exits cleanly.

    This function also supports single-task execution when max_tasks=1, which is
    how run_single_task() is implemented.

    Args:
        queue_dir: Override queue directory (for testing)
        max_tasks: Maximum tasks to process (None = unlimited)
        skip_resume: Skip incomplete work and start with new tasks
        dry_run: If True, only show what would run
        check_interval: Seconds between queue checks when nothing to run
        install_signal_handlers: Whether to install SIGINT/SIGTERM handlers (default True)
        shutdown_coordinator: Optional pre-existing coordinator (if None, gets global one)

    Returns:
        When max_tasks=1: Workflow result dict, or None if nothing to run
        When max_tasks!=1: Always returns None (runs until stopped)
    """
    from core.utils.async_http_client import cleanup_all_clients

    # Get or create shutdown coordinator
    coordinator = shutdown_coordinator or get_shutdown_coordinator()
    if install_signal_handlers:
        coordinator.install_signal_handlers()

    queue_manager = TaskQueueManager(queue_dir=queue_dir)
    checkpoint_mgr = CheckpointManager(queue_dir=queue_dir)
    budget_tracker = BudgetTracker(queue_dir=queue_dir)

    # Clean up any orphaned temp files from interrupted writes
    await checkpoint_mgr.cleanup_orphaned_temps()

    # Also clean up orphaned temp files in incremental directory
    incremental_mgr = IncrementalStateManager(incremental_dir=queue_dir / "incremental" if queue_dir else None)
    cleaned = await incremental_mgr.cleanup_orphaned_temps()
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} orphaned incremental temp files")

    tasks_processed = 0
    last_result: Optional[dict] = None  # Track result for max_tasks=1 case
    is_single_task_mode = max_tasks == 1

    try:
        while not coordinator.shutdown_requested:
            # Check for incomplete work first (unless skip_resume is set)
            if not skip_resume:
                incomplete = await checkpoint_mgr.get_incomplete_work()
                if incomplete:
                    checkpoint = incomplete[0]
                    task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")  # backward compat
                    task = queue_manager.get_task(task_id)

                    if task:
                        task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
                        task_identifier = task.get("topic") or task.get("query", "unknown")
                        logger.info(f"Resuming incomplete work: {task_id[:8]} ({task_type})")

                        if dry_run:
                            logger.info(f"Would resume: {task_identifier[:50]}...")
                            return None

                        last_result = await run_task_workflow(
                            task,
                            queue_manager,
                            checkpoint_mgr,
                            budget_tracker,
                            resume_from=checkpoint,
                            shutdown_coordinator=coordinator,
                        )
                        tasks_processed += 1

                        # Check for shutdown after task completion
                        if coordinator.shutdown_requested:
                            logger.info("Shutdown requested, exiting after task completion")
                            break

                        if max_tasks and tasks_processed >= max_tasks:
                            logger.info(f"Processed {tasks_processed} tasks, stopping")
                            break
                        continue

            # First check for bypass tasks (ignore concurrency limits)
            task = _find_bypass_task(queue_manager)
            if task:
                task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
                logger.info(f"Found bypass task: {task_type} (ignoring concurrency limits)")
            else:
                # Get next eligible task (respects concurrency)
                task = queue_manager.get_next_eligible_task()

            if not task:
                # In single-task mode, return immediately when no tasks
                if is_single_task_mode:
                    logger.info("No eligible tasks to run")
                    return None

                logger.info(f"No eligible tasks. Checking again in {check_interval}s...")
                # Use interruptible sleep - returns True if shutdown requested
                if await coordinator.wait_or_shutdown(check_interval):
                    logger.info("Shutdown requested during idle wait")
                    break
                continue

            task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
            task_identifier = task.get("topic") or task.get("query", "unknown")

            # Check if zero-cost workflow (skip budget check)
            workflow = get_workflow(task_type)
            is_zero_cost = getattr(workflow, "is_zero_cost", False)

            if not is_zero_cost:
                # Check budget for workflows that incur costs
                should_proceed, reason = budget_tracker.should_proceed()
                if not should_proceed:
                    # In single-task mode, return None on budget exceeded
                    if is_single_task_mode:
                        logger.warning(f"Cannot proceed: {reason}")
                        return None

                    logger.warning(f"Pausing due to budget: {reason}")
                    # Use interruptible sleep for an hour
                    if await coordinator.wait_or_shutdown(3600):
                        logger.info("Shutdown requested during budget pause")
                        break
                    continue
            else:
                logger.info(f"Skipping budget check for zero-cost workflow: {task_type}")

            if dry_run:
                logger.info(f"Would run ({task_type}): {task_identifier[:50]}...")
                return None if is_single_task_mode else None

            # Run the task
            try:
                last_result = await run_task_workflow(
                    task,
                    queue_manager,
                    checkpoint_mgr,
                    budget_tracker,
                    shutdown_coordinator=coordinator,
                )
                tasks_processed += 1
            except Exception as e:
                logger.error(f"Task failed: {e}")
                # In single-task mode, re-raise exceptions
                if is_single_task_mode:
                    raise
                # In loop mode, continue to next task

            # Check for shutdown after task completion
            if coordinator.shutdown_requested:
                logger.info("Shutdown requested, exiting after task completion")
                break

            if max_tasks and tasks_processed >= max_tasks:
                logger.info(f"Processed {tasks_processed} tasks, stopping")
                break

            # Get adaptive delay based on budget (skip for zero-cost workflows)
            # Only apply stagger delays in loop mode, not single-task mode
            if not is_zero_cost and not is_single_task_mode:
                config = queue_manager.get_concurrency_config()
                if config["mode"] == "stagger_hours":
                    base_hours = config["stagger_hours"]
                    adaptive_hours = budget_tracker.get_adaptive_stagger_hours(base_hours)
                    sleep_seconds = adaptive_hours * 3600

                    logger.info(
                        f"Sleeping {adaptive_hours:.1f} hours before next task (base: {base_hours}h, budget-adjusted)"
                    )
                    # Use interruptible sleep
                    if await coordinator.wait_or_shutdown(sleep_seconds):
                        logger.info("Shutdown requested during stagger sleep")
                        break

    finally:
        # Cleanup on exit
        logger.info("Cleaning up resources...")
        if install_signal_handlers:
            coordinator.remove_signal_handlers()
        await cleanup_all_clients()
        logger.info(f"Queue loop exiting. Processed {tasks_processed} tasks.")

    # Return result for single-task mode
    return last_result if is_single_task_mode else None


async def print_status_async():
    """Print current queue and budget status (async implementation)."""
    queue_manager = TaskQueueManager()
    checkpoint_mgr = CheckpointManager()
    budget_tracker = BudgetTracker()

    # Budget status
    print("=== BUDGET ===")
    status = budget_tracker.get_budget_status()
    print(f"  Month-to-date: {format_cost(status['current_cost'])}")
    print(f"  Budget: {format_cost(status['monthly_budget'])}")
    print(f"  Used: {status['percent_used']:.1f}%")
    print(f"  Action: {status['action']}")

    # Queue status
    print("\n=== QUEUE ===")
    stats = queue_manager.get_queue_stats()
    for status_name, count in stats["by_status"].items():
        print(f"  {status_name}: {count}")

    # Active work
    active = await checkpoint_mgr.get_active_work()
    if active:
        print(f"\n=== ACTIVE ({len(active)}) ===")
        for cp in active:
            task_id = cp.get("task_id") or cp.get("topic_id", "unknown")
            task_type = cp.get("task_type", DEFAULT_WORKFLOW_TYPE)
            print(f"  {task_id[:8]} ({task_type}): {cp['phase']}")

    # Next eligible
    next_task = queue_manager.get_next_eligible_task()
    if next_task:
        print("\n=== NEXT ===")
        task_type = next_task.get("task_type", DEFAULT_WORKFLOW_TYPE)
        task_identifier = next_task.get("topic") or next_task.get("query", "unknown")
        print(f"  [{task_type}] {task_identifier[:50]}...")


def print_status():
    """Print current queue and budget status.

    Sync wrapper that runs the async implementation.
    """
    import asyncio

    asyncio.run(print_status_async())
