"""
Task queue manager with safe concurrent access.

Provides:
- File locking via fcntl for cross-process coordination
- Atomic writes via temp file + rename
- Two-queue model: research_tasks + publish_tasks
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .categories import get_default_categories
from .paths import PUBLICATIONS_FILE, QUEUE_DIR
from .persistence import PUBLISH_TASK_TYPES, QueuePersistence
from .schemas import (
    Task,
    TaskCategory,
    TaskPriority,
    TaskQueue,
    TaskStatus,
)
from .workflows import DEFAULT_WORKFLOW_TYPE

logger = logging.getLogger(__name__)

# Fields that may be mutated via update_task().
# Identity / creation-time fields (id, task_type, created_at) are excluded
# so callers cannot accidentally overwrite them or inject arbitrary keys.
_MUTABLE_TASK_FIELDS: frozenset[str] = frozenset({
    # Lifecycle
    "status",
    "started_at",
    "completed_at",
    "error_message",
    "current_phase",
    "langsmith_run_id",
    # Scheduling
    "priority",
    "quality",
    "category",
    "next_run_after",
    "not_before",
    # Content (editable before execution)
    "topic",
    "query",
    "research_questions",
    "language",
    "date_range",
    # Metadata
    "notes",
    "tags",
    # Publish-task progress
    "items",
    "source_task_id",
    "manifest_path",
})


def _find_task_in_queue(queue: TaskQueue, task_id: str) -> tuple[Task | None, str | None]:
    """Find a task in either queue array.

    Returns:
        (task, array_key) where array_key is "research_tasks" or "publish_tasks",
        or (None, None) if not found.
    """
    for key in ("research_tasks", "publish_tasks"):
        for task in queue[key]:
            if task["id"] == task_id:
                return task, key
    return None, None


class TaskQueueManager:
    """Manages the task queue with safe concurrent access."""

    def __init__(self, queue_dir: Path | None = None):
        """Initialize the queue manager.

        Args:
            queue_dir: Override queue directory (for testing)
        """
        self.queue_dir = queue_dir or QUEUE_DIR
        self.queue_file = self.queue_dir / "queue.json"
        self.lock_file = self.queue_dir / "queue.lock"
        self.publications_file = self.queue_dir / "publications.json"

        # Initialize persistence
        self.persistence = QueuePersistence(self.queue_file, self.lock_file)

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_queue_exists()

    def _ensure_queue_exists(self) -> None:
        """Create queue file if it doesn't exist."""
        if not self.queue_file.exists():
            # Use PUBLICATIONS_FILE for default categories (global source of truth)
            default_categories = get_default_categories(PUBLICATIONS_FILE)

            initial_queue: TaskQueue = {
                "version": "2.0",
                "categories": default_categories,
                "last_category_index": -1,
                "research_tasks": [],
                "publish_tasks": [],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence.write_queue(initial_queue)

    def add_task(
        self,
        category: TaskCategory | str,
        priority: TaskPriority | int = TaskPriority.NORMAL,
        quality: str = "standard",
        notes: str | None = None,
        tags: list[str] | None = None,
        task_type: str = DEFAULT_WORKFLOW_TYPE,
        # Type-specific fields (passed through)
        topic: str | None = None,  # lit_review_full
        research_questions: list[str] | None = None,  # lit_review_full
        language: str | None = None,  # both
        date_range: tuple[int, int] | None = None,  # lit_review_full
        query: str | None = None,  # web_research
        **kwargs,  # Future extensibility
    ) -> str:
        """Add a task to the queue.

        Routes to research_tasks or publish_tasks based on task_type.

        Returns:
            Task ID (UUID string)
        """
        task_id = str(uuid.uuid4())

        # Normalize category/priority to values
        cat_value = category.value if isinstance(category, TaskCategory) else category
        pri_value = priority.value if isinstance(priority, TaskPriority) else priority

        # Build base task fields (common to all types)
        base_fields = {
            "id": task_id,
            "task_type": task_type,
            "category": cat_value,
            "priority": pri_value,
            "status": TaskStatus.PENDING.value,
            "quality": quality,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "langsmith_run_id": None,
            "current_phase": None,
            "error_message": None,
            "notes": notes,
            "tags": tags or [],
        }

        # Add type-specific fields
        if task_type == "lit_review_full":
            new_task: Task = {
                **base_fields,
                "topic": topic or "",
                "research_questions": research_questions,
                "language": language or "en",
                "date_range": date_range,
            }
            identifier = topic or ""
        elif task_type == "web_research":
            new_task: Task = {
                **base_fields,
                "query": query or "",
                "language": language,  # Optional for web research
            }
            identifier = query or ""
        elif task_type == "illustrate_and_export":
            new_task: Task = {
                **base_fields,
                "topic": topic or "",
                "source_task_id": kwargs.get("source_task_id", ""),
                "manifest_path": kwargs.get("manifest_path", ""),
                "items": kwargs.get("items", []),
                "not_before": kwargs.get("not_before"),
                "next_run_after": None,
            }
            identifier = topic or ""
        else:
            # Generic fallback - include all provided fields
            new_task: Task = {
                **base_fields,
                "topic": topic,
                "query": query,
                "language": language,
                **kwargs,
            }
            identifier = topic or query or ""

        # Route to correct array
        array_key = "publish_tasks" if task_type in PUBLISH_TASK_TYPES else "research_tasks"

        with self.persistence.lock():
            queue = self.persistence.read_queue()
            queue[array_key].append(new_task)
            self.persistence.write_queue(queue)

        logger.info(f"Added task {task_id} ({task_type}) to {array_key}: {identifier[:50]}...")
        return task_id

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID (searches both queues)."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            task, _ = _find_task_in_queue(queue, task_id)
            return task

    def mark_started(self, task_id: str, langsmith_run_id: str) -> None:
        """Mark task as started."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            task, _ = _find_task_in_queue(queue, task_id)
            if task:
                task["status"] = TaskStatus.IN_PROGRESS.value
                task["started_at"] = datetime.now(timezone.utc).isoformat()
                task["langsmith_run_id"] = langsmith_run_id
                self.persistence.write_queue(queue)

    def mark_completed(self, task_id: str) -> None:
        """Mark task as completed."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            task, _ = _find_task_in_queue(queue, task_id)
            if task:
                task["status"] = TaskStatus.COMPLETED.value
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
                self.persistence.write_queue(queue)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            task, _ = _find_task_in_queue(queue, task_id)
            if task:
                task["status"] = TaskStatus.FAILED.value
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
                task["error_message"] = error
                self.persistence.write_queue(queue)

    def update_phase(self, task_id: str, phase: str) -> None:
        """Update current workflow phase for checkpointing."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            task, _ = _find_task_in_queue(queue, task_id)
            if task:
                task["current_phase"] = phase
                self.persistence.write_queue(queue)

    def list_tasks(
        self,
        status: TaskStatus | str | None = None,
        category: TaskCategory | str | None = None,
    ) -> list[Task]:
        """List tasks with optional filtering (searches both queues)."""
        # Normalize to values
        status_val = status.value if isinstance(status, TaskStatus) else status
        cat_val = category.value if isinstance(category, TaskCategory) else category

        with self.persistence.lock():
            queue = self.persistence.read_queue()
            tasks = queue["research_tasks"] + queue["publish_tasks"]

            if status_val:
                tasks = [t for t in tasks if t["status"] == status_val]
            if cat_val:
                tasks = [t for t in tasks if t["category"] == cat_val]

            return tasks

    def reorder(self, task_ids: list[str]) -> None:
        """Reorder research tasks by providing list of task IDs in desired order.

        Only operates on research_tasks (publish tasks use priority + not_before).
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()

            # Build lookup
            task_map = {t["id"]: t for t in queue["research_tasks"]}

            # Reorder
            reordered = [task_map[tid] for tid in task_ids if tid in task_map]

            # Add any tasks not in the list (safety)
            seen = set(task_ids)
            for task in queue["research_tasks"]:
                if task["id"] not in seen:
                    reordered.append(task)

            queue["research_tasks"] = reordered
            self.persistence.write_queue(queue)

    def get_categories(self) -> list[str]:
        """Get list of categories."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            return queue["categories"]

    def set_categories(self, categories: list[str]) -> None:
        """Update category list."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            queue["categories"] = categories
            queue["last_category_index"] = -1
            self.persistence.write_queue(queue)

    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            all_tasks = queue["research_tasks"] + queue["publish_tasks"]

            by_status = {}
            by_category = {}

            for task in all_tasks:
                status = task["status"]
                by_status[status] = by_status.get(status, 0) + 1

                category = task["category"]
                by_category[category] = by_category.get(category, 0) + 1

            return {
                "total": len(all_tasks),
                "by_status": by_status,
                "by_category": by_category,
                "research_count": len(queue["research_tasks"]),
                "publish_count": len(queue["publish_tasks"]),
            }

    def update_task(self, task_id: str, **updates) -> bool:
        """Update mutable fields on a task (searches both queues).

        Only fields listed in ``_MUTABLE_TASK_FIELDS`` are accepted.
        Protected fields (id, task_type, created_at) and arbitrary keys
        are silently dropped to prevent accidental overwrites or injection.
        """
        filtered = {k: v for k, v in updates.items() if k in _MUTABLE_TASK_FIELDS}
        if not filtered:
            return False
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            task, _ = _find_task_in_queue(queue, task_id)
            if task:
                task.update(filtered)
                self.persistence.write_queue(queue)
                return True
        return False
