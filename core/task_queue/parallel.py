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
from .categories import load_categories_from_publications
from .checkpoint import CheckpointManager
from .lifecycle import cleanup_supervisor_resources
from .paths import PUBLICATIONS_FILE
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

    # Identify resumable tasks (have checkpoint but owning process is dead)
    incomplete = await checkpoint_mgr.get_incomplete_work()
    resumable_ids = {cp["task_id"] for cp in incomplete}
    if resumable_ids:
        logger.info(f"Found {len(resumable_ids)} resumable task(s) with checkpoints")
    checkpoints_by_id = {cp["task_id"]: cp for cp in incomplete}

    # Atomic task selection (sync I/O with fcntl.flock — run in thread pool)
    tasks = await asyncio.to_thread(
        _select_tasks, queue_manager, count, resumable_ids
    )
    if not tasks:
        logger.info("No eligible tasks to run")
        coordinator.remove_signal_handlers()
        return []

    logger.info(f"Selected {len(tasks)} tasks for parallel execution")
    selected_ids = {task["id"] for task in tasks}
    for task in tasks:
        tid = task["id"][:8]
        identifier = task.get("topic") or task.get("query", "unknown")
        resuming = " (resuming)" if task["id"] in resumable_ids else ""
        logger.info(f"  {tid}: {identifier[:60]}{resuming}")

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
                resume_from=checkpoints_by_id.get(task["id"]),
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


def _select_tasks(
    queue_manager: TaskQueueManager,
    count: int,
    resumable_ids: set[str] | None = None,
) -> list[Task]:
    """Atomically select and claim tasks under queue lock.

    Called via asyncio.to_thread() to avoid blocking the event loop
    with fcntl.flock().

    Prioritises resumable tasks (those with checkpoints in current_work.json)
    over fresh pending tasks. Resets orphaned IN_PROGRESS tasks (no checkpoint)
    back to PENDING. Uses round-robin category rotation (matching the
    sequential scheduler) to ensure thematic diversity. Within each category,
    resumable tasks come first, then highest-priority, then FIFO.
    """
    resumable_ids = resumable_ids or set()

    with queue_manager.persistence.lock():
        queue = queue_manager.persistence.read_queue()

        # Reset orphaned IN_PROGRESS tasks (no checkpoint) back to PENDING
        for task in queue["topics"]:
            if (
                task["status"] == TaskStatus.IN_PROGRESS.value
                and task["id"] not in resumable_ids
            ):
                task["status"] = TaskStatus.PENDING.value
                task.pop("started_at", None)

        # Build candidate pool: resumable (still IN_PROGRESS) + PENDING
        candidates = [
            t
            for t in queue["topics"]
            if t["status"] in (TaskStatus.IN_PROGRESS.value, TaskStatus.PENDING.value)
        ]
        if not candidates:
            return []

        categories = load_categories_from_publications(PUBLICATIONS_FILE)
        last_idx = queue.get("last_category_index", -1)

        # Sync categories if they've changed
        if queue.get("categories") != categories:
            queue["categories"] = categories
            if last_idx >= len(categories):
                last_idx = -1

        # Build per-category task lists.
        # Sort: resumable first, then priority desc, then FIFO.
        by_category: dict[str, list[Task]] = {}
        for task in candidates:
            by_category.setdefault(task["category"], []).append(task)
        for tasks in by_category.values():
            tasks.sort(
                key=lambda t: (
                    0 if t["id"] in resumable_ids else 1,
                    -t.get("priority", 2),
                    t.get("created_at", ""),
                )
            )

        # Walk categories round-robin, picking one task per category per pass.
        # start_idx is fixed for the entire inner loop so that updating
        # last_idx on selection doesn't shift the walk mid-iteration.
        selected: list[Task] = []
        start_idx = last_idx + 1
        passes = 0
        while len(selected) < count and passes < len(candidates):
            added_this_round = False
            for offset in range(len(categories)):
                if len(selected) >= count:
                    break
                cat_idx = (start_idx + offset) % len(categories)
                category = categories[cat_idx]
                cat_tasks = by_category.get(category, [])
                if cat_tasks:
                    selected.append(cat_tasks.pop(0))
                    last_idx = cat_idx
                    added_this_round = True
            if not added_this_round:
                break
            start_idx = last_idx + 1
            passes += 1

        # Fallback: if categories don't cover all tasks (e.g. deprecated
        # category), fill remaining slots with highest-priority leftovers
        if len(selected) < count:
            selected_set = {t["id"] for t in selected}
            leftovers = [t for t in candidates if t["id"] not in selected_set]
            leftovers.sort(
                key=lambda t: (
                    0 if t["id"] in resumable_ids else 1,
                    -t.get("priority", 2),
                    t.get("created_at", ""),
                )
            )
            selected.extend(leftovers[: count - len(selected)])

        # Reset unselected resumable tasks back to PENDING
        selected_set = {t["id"] for t in selected}
        for task in queue["topics"]:
            if (
                task["status"] == TaskStatus.IN_PROGRESS.value
                and task["id"] not in selected_set
            ):
                task["status"] = TaskStatus.PENDING.value
                task.pop("started_at", None)

        # Persist category index and mark newly selected tasks as in-progress
        queue["last_category_index"] = last_idx
        now = datetime.now(timezone.utc).isoformat()
        for task in selected:
            if task["status"] == TaskStatus.PENDING.value:
                task["status"] = TaskStatus.IN_PROGRESS.value
                task["started_at"] = now
        queue_manager.persistence.write_queue(queue)
    return selected
