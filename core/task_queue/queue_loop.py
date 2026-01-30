"""
Queue processing loop with graceful shutdown.

Provides:
- Continuous queue processing
- Single task execution
- Resume incomplete work
- Budget-aware delays
- Signal handling for graceful shutdown
"""

import logging
from pathlib import Path
from typing import Optional

from .budget_tracker import BudgetTracker
from .checkpoint import CheckpointManager
from .incremental_state import IncrementalStateManager
from .queue_manager import TaskQueueManager
from .shutdown import ShutdownCoordinator, get_shutdown_coordinator
from .task_selector import _find_bypass_task
from .workflow_executor import run_task_workflow
from .workflows import get_workflow, DEFAULT_WORKFLOW_TYPE

logger = logging.getLogger(__name__)


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
