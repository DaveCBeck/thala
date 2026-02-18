"""Parallel workflow supervisor.

Runs multiple task workflows concurrently via asyncio.gather().
Two-queue selection: publish tasks (date-gated) take priority,
remaining slots filled by research tasks (category round-robin).
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


def _reset_task_to_pending(task: dict) -> None:
    """Reset a task to PENDING, clearing all stale run metadata."""
    task["status"] = TaskStatus.PENDING.value
    task.pop("started_at", None)
    task.pop("error_message", None)
    task.pop("current_phase", None)


async def run_parallel_tasks(
    count: int = 5,
    stagger_minutes: float = 3.0,
    queue_dir: Path | None = None,
    _manage_signals: bool = True,
) -> list[dict | BaseException]:
    """Run multiple tasks concurrently via asyncio.gather().

    Args:
        count: Maximum number of tasks to run concurrently.
        stagger_minutes: Minutes between each workflow start.
        queue_dir: Override queue directory (for testing).
        _manage_signals: Whether to install/remove signal handlers.
            Set to False when called from run_daemon_loop(), which
            manages signal handler lifecycle itself.

    Returns:
        List of workflow results or exceptions, one per task.
    """
    coordinator = get_shutdown_coordinator()
    if _manage_signals:
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
    tasks = await asyncio.to_thread(_select_tasks, queue_manager, count, resumable_ids)
    if not tasks:
        logger.info("No eligible tasks to run")
        if _manage_signals:
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
            should_proceed, reason = await asyncio.to_thread(budget_tracker.should_proceed)
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
        if _manage_signals:
            coordinator.remove_signal_handlers()
        await cleanup_supervisor_resources()

        # Reset orphaned IN_PROGRESS tasks back to PENDING.
        # Uses persistence directly for atomic batch update (see _select_tasks
        # docstring re: TODO #150 for rationale).
        try:

            def _reset_orphaned():
                with queue_manager.persistence.lock():
                    queue = queue_manager.persistence.read_queue()
                    reset_count = 0
                    for key in ("research_tasks", "publish_tasks"):
                        for task in queue[key]:
                            if task["id"] in selected_ids and task["status"] == TaskStatus.IN_PROGRESS.value:
                                _reset_task_to_pending(task)
                                reset_count += 1
                    if reset_count:
                        queue_manager.persistence.write_queue(queue)
                        logger.info(f"Reset {reset_count} orphaned task(s) back to PENDING")

            await asyncio.to_thread(_reset_orphaned)
        except Exception:
            logger.exception("Error resetting orphaned tasks")


async def run_daemon_loop(
    count: int = 5,
    stagger_minutes: float = 3.0,
    check_interval: float = 300.0,
    max_batches: int | None = None,
    queue_dir: Path | None = None,
) -> None:
    """Run parallel dispatcher in a loop for daemon mode.

    Args:
        count: Tasks per batch.
        stagger_minutes: Minutes between workflow starts within a batch.
        check_interval: Seconds between batch runs when idle.
        max_batches: Stop after this many batches (None = unlimited).
        queue_dir: Override queue directory (for testing).
    """
    coordinator = get_shutdown_coordinator()
    coordinator.install_signal_handlers()

    batches_run = 0

    try:
        while not coordinator.shutdown_requested:
            results = await run_parallel_tasks(
                count=count,
                stagger_minutes=stagger_minutes,
                queue_dir=queue_dir,
                _manage_signals=False,
            )

            batches_run += 1
            if max_batches and batches_run >= max_batches:
                logger.info(f"Reached max batches ({max_batches}), stopping")
                break

            if coordinator.shutdown_requested:
                break

            # Wait before next batch
            logger.info(f"Batch complete ({len(results)} tasks). Next check in {check_interval}s")
            if await coordinator.wait_or_shutdown(check_interval):
                logger.info("Shutdown requested during idle wait")
                break
    finally:
        coordinator.remove_signal_handlers()


def _select_tasks(
    queue_manager: TaskQueueManager,
    count: int,
    resumable_ids: set[str] | None = None,
) -> list[Task]:
    """Atomically select and claim tasks under queue lock.

    Called via asyncio.to_thread() to avoid blocking the event loop
    with fcntl.flock().

    Selection order:
    1. Publish tasks (date-gated by not_before, DEFERRED by next_run_after)
    2. Research tasks (category round-robin, resumable first, then priority+FIFO)

    NOTE -- Intentional persistence-layer bypass (see TODO #150):
        This function accesses queue_manager.persistence (lock / read_queue /
        write_queue) directly instead of going through TaskQueueManager's
        public methods (add_task, mark_started, list_tasks, etc.).

        The reason is atomicity: we must hold a single lock while reading the
        full queue, evaluating eligibility across *both* task lists, resetting
        orphaned tasks, marking multiple winners as IN_PROGRESS, updating the
        category round-robin cursor, and writing everything back in one shot.

        TaskQueueManager's API is designed for single-task CRUD -- each method
        independently acquires the lock, reads, mutates one task, and writes.
        Calling N separate methods here would mean N lock-acquire/release
        cycles with no guarantee that the queue state is consistent between
        them (another process could interleave).

        The same pattern applies to _reset_orphaned() in the finally block of
        run_parallel_tasks() for the same atomic-batch-update reason.
    """
    resumable_ids = resumable_ids or set()

    with queue_manager.persistence.lock():
        queue = queue_manager.persistence.read_queue()
        now = datetime.now(timezone.utc)

        # Reset orphaned IN_PROGRESS tasks (no checkpoint) back to PENDING
        for key in ("research_tasks", "publish_tasks"):
            for task in queue[key]:
                if task["status"] == TaskStatus.IN_PROGRESS.value and task["id"] not in resumable_ids:
                    _reset_task_to_pending(task)

        # Phase 1: select publish tasks
        selected = _select_publish_tasks(queue["publish_tasks"], count, now, resumable_ids)

        # Phase 2: fill remaining slots with research tasks
        remaining = count - len(selected)
        if remaining > 0:
            categories = load_categories_from_publications(PUBLICATIONS_FILE)
            last_idx = queue.get("last_category_index", -1)

            # Sync categories if they've changed
            if queue.get("categories") != categories:
                queue["categories"] = categories
                if last_idx >= len(categories):
                    last_idx = -1

            research_selected, last_idx = _select_research_tasks(
                queue["research_tasks"],
                remaining,
                categories,
                last_idx,
                resumable_ids,
                now,
            )
            selected.extend(research_selected)
            queue["last_category_index"] = last_idx

        if not selected:
            queue_manager.persistence.write_queue(queue)
            return []

        # Reset unselected resumable tasks back to PENDING
        selected_set = {t["id"] for t in selected}
        for key in ("research_tasks", "publish_tasks"):
            for task in queue[key]:
                if task["status"] == TaskStatus.IN_PROGRESS.value and task["id"] not in selected_set:
                    _reset_task_to_pending(task)

        # Mark selected tasks as in-progress
        now_iso = datetime.now(timezone.utc).isoformat()
        for task in selected:
            if task["status"] in (TaskStatus.PENDING.value, TaskStatus.DEFERRED.value):
                task["status"] = TaskStatus.IN_PROGRESS.value
                task["started_at"] = now_iso

        queue_manager.persistence.write_queue(queue)
    return selected


def _task_sort_key(t: Task, resumable_ids: set[str]) -> tuple:
    """Sort key: resumable first, then priority desc, then FIFO."""
    return (
        0 if t["id"] in resumable_ids else 1,
        -t.get("priority", 2),
        t.get("created_at", ""),
    )


def _select_publish_tasks(
    publish_tasks: list[Task],
    count: int,
    now: datetime,
    resumable_ids: set[str],
) -> list[Task]:
    """Select eligible publish tasks.

    PENDING tasks: include if not_before is absent or not_before <= now.
    DEFERRED tasks: include if next_run_after <= now (ignore not_before).
    IN_PROGRESS tasks in resumable_ids: include (resuming).

    Sort by priority desc, then created_at asc.
    """
    candidates = []
    for t in publish_tasks:
        if t["status"] == TaskStatus.IN_PROGRESS.value and t["id"] in resumable_ids:
            candidates.append(t)
        elif t["status"] == TaskStatus.PENDING.value:
            not_before = t.get("not_before")
            if not not_before or _is_past(not_before, now):
                candidates.append(t)
        elif t["status"] == TaskStatus.DEFERRED.value:
            next_run = t.get("next_run_after")
            if not next_run or _is_past(next_run, now):
                candidates.append(t)

    # Sort: resumable first, then priority desc, then FIFO
    candidates.sort(key=lambda t: _task_sort_key(t, resumable_ids))
    return candidates[:count]


def _select_research_tasks(
    research_tasks: list[Task],
    count: int,
    categories: list[str],
    last_idx: int,
    resumable_ids: set[str],
    now: datetime,
) -> tuple[list[Task], int]:
    """Select research tasks via category round-robin.

    Returns (selected_tasks, updated_last_category_index).
    """
    # Build candidate pool
    candidates = []
    for t in research_tasks:
        if t["status"] in (TaskStatus.IN_PROGRESS.value, TaskStatus.PENDING.value):
            candidates.append(t)
        elif t["status"] == TaskStatus.DEFERRED.value:
            next_run = t.get("next_run_after")
            if not next_run or _is_past(next_run, now):
                candidates.append(t)

    if not candidates:
        return [], last_idx

    # Build per-category task lists
    # Map task categories to canonical form (title-case from publications.json)
    cat_canonical = {c.lower(): c for c in categories}
    by_category: dict[str, list[Task]] = {}
    for task in candidates:
        canon = cat_canonical.get(task["category"].lower(), task["category"])
        by_category.setdefault(canon, []).append(task)
    for tasks in by_category.values():
        tasks.sort(key=lambda t: _task_sort_key(t, resumable_ids))

    # Walk categories round-robin
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

    # Fallback: fill remaining slots with highest-priority leftovers
    if len(selected) < count:
        selected_set = {t["id"] for t in selected}
        leftovers = [t for t in candidates if t["id"] not in selected_set]
        leftovers.sort(key=lambda t: _task_sort_key(t, resumable_ids))
        selected.extend(leftovers[: count - len(selected)])

    return selected, last_idx


def _is_past(iso_str: str, now: datetime) -> bool:
    """Check if an ISO datetime string is in the past (or malformed)."""
    try:
        return now >= datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        logger.warning(f"Malformed datetime {iso_str!r}, treating as eligible")
        return True
