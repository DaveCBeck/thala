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
from typing import Optional

from .budget_tracker import BudgetTracker
from .checkpoint import CheckpointManager
from .queue_manager import TaskQueueManager
from .schemas import Task, TaskStatus
from .shutdown import get_shutdown_coordinator
from .workflow_executor import run_task_workflow

logger = logging.getLogger(__name__)


async def run_parallel_tasks(
    count: int = 5,
    stagger_minutes: float = 3.0,
    queue_dir: Optional[Path] = None,
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
                await asyncio.sleep(delay)
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
        # Supervisor owns broker lifecycle
        from core.llm_broker import get_broker, is_broker_enabled

        if is_broker_enabled():
            try:
                broker = get_broker()
                if broker._started:
                    await broker.stop()
            except Exception:
                logger.exception("Error stopping broker")

        coordinator.remove_signal_handlers()
        from core.utils.async_http_client import cleanup_all_clients

        await cleanup_all_clients()


def _select_tasks(queue_manager: TaskQueueManager, count: int) -> list[Task]:
    """Atomically select and claim pending tasks under queue lock.

    Called via asyncio.to_thread() to avoid blocking the event loop
    with fcntl.flock().
    """
    with queue_manager.persistence.lock():
        queue = queue_manager.persistence.read_queue()
        pending = [t for t in queue["topics"] if t["status"] == TaskStatus.PENDING.value]
        selected = pending[:count]
        now = datetime.now(timezone.utc).isoformat()
        for task in selected:
            task["status"] = TaskStatus.IN_PROGRESS.value
            task["started_at"] = now
        queue_manager.persistence.write_queue(queue)
    return selected
