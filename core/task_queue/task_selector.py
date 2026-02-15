"""
Task selection logic for bypassing concurrency limits.

Provides:
- Find tasks that bypass concurrency controls
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .queue_manager import TaskQueueManager
from .schemas import Task, TaskStatus
from .workflows import get_workflow

logger = logging.getLogger(__name__)


def _is_deferred_eligible(task: Task, now: datetime) -> bool:
    """Check whether a DEFERRED task is past its next_run_after time."""
    next_run = task.get("next_run_after")
    if not next_run:
        return True
    try:
        return now >= datetime.fromisoformat(next_run)
    except (ValueError, TypeError):
        return True


def _find_bypass_task(queue_manager: TaskQueueManager) -> Optional[Task]:
    """Find a pending task that bypasses concurrency limits.

    Bypass tasks (like publish_series) can run regardless of stagger_hours
    or max_concurrent settings.

    Args:
        queue_manager: Queue manager instance

    Returns:
        First eligible bypass task, or None
    """
    pending_tasks = queue_manager.list_tasks(status=TaskStatus.PENDING)
    deferred_tasks = queue_manager.list_tasks(status=TaskStatus.DEFERRED)

    now = datetime.now(timezone.utc)
    candidates = pending_tasks + [t for t in deferred_tasks if _is_deferred_eligible(t, now)]

    for task in candidates:
        task_type = task.get("task_type", "lit_review_full")
        workflow = get_workflow(task_type)
        if getattr(workflow, "bypass_concurrency", False):
            return task

    return None
