"""Parallel workflow supervisor.

Runs multiple task workflows concurrently via asyncio.gather(),
bypassing the sequential queue loop. Manages broker lifecycle
at supervisor level to avoid killing the shared singleton
when individual workflows complete.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from .budget_tracker import BudgetTracker
from .checkpoint import CheckpointManager
from .lifecycle import cleanup_supervisor_resources
from .queue_manager import TaskQueueManager
from .schemas import Task, TaskStatus
from .shutdown import get_shutdown_coordinator
from .workflow_executor import run_task_workflow

logger = logging.getLogger(__name__)


async def run_parallel_tasks(
    count: int = 5,
    stagger_minutes: float = 3.0,
    queue_dir: Path | None = None,
) -> list[dict | BaseException]:
    """Run multiple tasks concurrently via asyncio.gather().

    Bypasses the queue loop for parallel execution. Manages broker
    lifecycle at supervisor level.

    Args:
        count: Maximum number of tasks to run concurrently.
        stagger_minutes: Minutes between each workflow start.
        queue_dir: Override queue directory (for testing).

    Returns:
        List of workflow results or exceptions, one per task.
    """
    coordinator = get_shutdown_coordinator()
    coordinator.install_signal_handlers()
    queue_manager = TaskQueueManager(queue_dir=queue_dir)
    checkpoint_mgr = CheckpointManager(queue_dir=queue_dir)
    budget_tracker = BudgetTracker(queue_dir=queue_dir)

    # Atomic task selection (sync I/O with fcntl.flock — run in thread pool)
    tasks = await asyncio.to_thread(_select_tasks, queue_manager, count)
    if not tasks:
        logger.info("No eligible tasks to run")
        coordinator.remove_signal_handlers()
        return []

    logger.info(f"Selected {len(tasks)} tasks for parallel execution")
    selected_ids = {task["id"] for task in tasks}
    for task in tasks:
        tid = task["id"][:8]
        identifier = task.get("topic") or task.get("query", "unknown")
        logger.info(f"  {tid}: {identifier[:60]}")

    try:
        # Stagger starts and run concurrently
        async def run_with_stagger(task: Task, index: int) -> dict:
            if index > 0:
                delay = index * stagger_minutes * 60
                tid = task["id"][:8]
                logger.info(f"Task {tid}: starting in {delay:.0f}s")
                if await coordinator.wait_or_shutdown(delay):
                    logger.info(f"Task {tid}: skipped due to shutdown")
                    raise asyncio.CancelledError()

            # Check budget before launching workflow
            should_proceed, reason = budget_tracker.should_proceed()
            if not should_proceed:
                tid = task["id"][:8]
                logger.warning(f"Task {tid}: skipped due to budget: {reason}")
                return {"status": "skipped", "reason": f"budget: {reason}"}

            return await run_task_workflow(
                task,
                queue_manager,
                checkpoint_mgr,
                budget_tracker,
                shutdown_coordinator=coordinator,
            )

        results = await asyncio.gather(
            *[run_with_stagger(task, i) for i, task in enumerate(tasks)],
            return_exceptions=True,
        )

        # Log results
        for task, result in zip(tasks, results):
            tid = task["id"][:8]
            if isinstance(result, asyncio.CancelledError):
                logger.info(f"Task {tid} was cancelled")
            elif isinstance(result, BaseException):
                logger.error(f"Task {tid} failed: {result}")
            else:
                logger.info(f"Task {tid}: {result.get('status', 'unknown')}")

        return results

    finally:
        coordinator.remove_signal_handlers()
        await cleanup_supervisor_resources()

        # Reset orphaned IN_PROGRESS tasks back to PENDING so they are
        # retried on the next invocation instead of staying stuck.
        try:

            def _reset_orphaned():
                with queue_manager.persistence.lock():
                    queue = queue_manager.persistence.read_queue()
                    reset_count = 0
                    for task in queue["topics"]:
                        if (
                            task["id"] in selected_ids
                            and task["status"] == TaskStatus.IN_PROGRESS.value
                        ):
                            task["status"] = TaskStatus.PENDING.value
                            task.pop("started_at", None)
                            reset_count += 1
                    if reset_count:
                        queue_manager.persistence.write_queue(queue)
                        logger.info(
                            f"Reset {reset_count} orphaned task(s) back to PENDING"
                        )

            await asyncio.to_thread(_reset_orphaned)
        except Exception:
            logger.exception("Error resetting orphaned tasks")


def _select_tasks(queue_manager: TaskQueueManager, count: int) -> list[Task]:
    """Atomically select and claim pending tasks under queue lock.

    Called via asyncio.to_thread() to avoid blocking the event loop
    with fcntl.flock().

    Tasks are sorted by priority (highest first), then by creation time
    (oldest first) for FIFO within the same priority level. This mirrors
    the ordering used by TaskScheduler.get_next_eligible_task().
    """
    with queue_manager.persistence.lock():
        queue = queue_manager.persistence.read_queue()
        pending = [t for t in queue["topics"] if t["status"] == TaskStatus.PENDING.value]
        # Sort by priority descending (4=urgent > 3=high > 2=normal > 1=low),
        # then by created_at ascending (FIFO within same priority).
        pending.sort(key=lambda t: (-t.get("priority", 2), t.get("created_at", "")))
        selected = pending[:count]
        now = datetime.now(timezone.utc).isoformat()
        for task in selected:
            task["status"] = TaskStatus.IN_PROGRESS.value
            task["started_at"] = now
        queue_manager.persistence.write_queue(queue)
    return selected
