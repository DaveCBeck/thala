"""
Task queue manager with safe concurrent access.

Provides:
- File locking via fcntl for cross-process coordination
- Atomic writes via temp file + rename
- Round-robin category selection with priority within category
- Flexible concurrency control (max_concurrent or stagger_hours)
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .categories import get_default_categories
from .paths import PUBLICATIONS_FILE, QUEUE_DIR
from .persistence import QueuePersistence
from .publishing import PublishingScheduler
from .scheduler import TaskScheduler
from .schemas import (
    ConcurrencyConfig,
    Task,
    TaskCategory,
    TaskPriority,
    TaskQueue,
    TaskStatus,
)

# Default workflow type for backward compatibility
DEFAULT_WORKFLOW_TYPE = "lit_review_full"

logger = logging.getLogger(__name__)


class TaskQueueManager:
    """Manages the task queue with safe concurrent access."""

    def __init__(self, queue_dir: Optional[Path] = None):
        """Initialize the queue manager.

        Args:
            queue_dir: Override queue directory (for testing)
        """
        self.queue_dir = queue_dir or QUEUE_DIR
        self.queue_file = self.queue_dir / "queue.json"
        self.lock_file = self.queue_dir / "queue.lock"
        self.publications_file = self.queue_dir / "publications.json"

        # Initialize components
        self.persistence = QueuePersistence(self.queue_file, self.lock_file)
        self.scheduler = TaskScheduler(self.publications_file)
        self.publishing = PublishingScheduler()

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_queue_exists()

    def _ensure_queue_exists(self) -> None:
        """Create queue file if it doesn't exist."""
        if not self.queue_file.exists():
            # Use PUBLICATIONS_FILE for default categories (global source of truth)
            default_categories = get_default_categories(PUBLICATIONS_FILE)

            initial_queue: TaskQueue = {
                "version": "1.0",
                "concurrency": {
                    "mode": "stagger_hours",
                    "max_concurrent": 1,
                    "stagger_hours": 36.0,
                },
                "categories": default_categories,
                "last_category_index": -1,
                "topics": [],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence.write_queue(initial_queue)

    def add_task(
        self,
        category: TaskCategory | str,
        priority: TaskPriority | int = TaskPriority.NORMAL,
        quality: str = "standard",
        notes: Optional[str] = None,
        tags: Optional[list[str]] = None,
        task_type: str = DEFAULT_WORKFLOW_TYPE,
        # Type-specific fields (passed through)
        topic: Optional[str] = None,  # lit_review_full
        research_questions: Optional[list[str]] = None,  # lit_review_full
        language: Optional[str] = None,  # both
        date_range: Optional[tuple[int, int]] = None,  # lit_review_full
        query: Optional[str] = None,  # web_research
        **kwargs,  # Future extensibility
    ) -> str:
        """Add a task to the queue.

        Args:
            category: Thematic category
            priority: Task priority
            quality: Quality tier for workflow
            notes: User/LLM notes
            tags: Searchable tags
            task_type: Workflow type ("lit_review_full", "web_research", etc.)

            # Type-specific fields:
            topic: Main topic text (lit_review_full)
            research_questions: Optional pre-defined questions (lit_review_full)
            language: ISO 639-1 language code
            date_range: (start_year, end_year) for paper search (lit_review_full)
            query: Research query (web_research)

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

        with self.persistence.lock():
            queue = self.persistence.read_queue()
            queue["topics"].append(new_task)
            self.persistence.write_queue(queue)

        logger.info(f"Added task {task_id} ({task_type}): {identifier[:50]}...")
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: Task UUID

        Returns:
            Task if found, None otherwise
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    return task
        return None

    def get_next_eligible_task(self) -> Optional[Task]:
        """Get next task eligible to run.

        Selection considers:
        1. Concurrency limits (max_concurrent or stagger_hours)
        2. Category rotation (round-robin)
        3. Priority (within same category rotation)

        Returns:
            Next eligible task, or None if none eligible
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            selected = self.scheduler.get_next_eligible_task(queue)

            # Write back queue if scheduler modified it (category index update)
            if selected:
                self.persistence.write_queue(queue)

            return selected

    def mark_started(self, task_id: str, langsmith_run_id: str) -> None:
        """Mark task as started.

        Args:
            task_id: Task UUID
            langsmith_run_id: LangSmith run ID for cost tracking
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["status"] = TaskStatus.IN_PROGRESS.value
                    task["started_at"] = datetime.now(timezone.utc).isoformat()
                    task["langsmith_run_id"] = langsmith_run_id
                    break
            self.persistence.write_queue(queue)

    def mark_completed(self, task_id: str) -> None:
        """Mark task as completed."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["status"] = TaskStatus.COMPLETED.value
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self.persistence.write_queue(queue)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed.

        Args:
            task_id: Task UUID
            error: Error message
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["status"] = TaskStatus.FAILED.value
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    task["error_message"] = error
                    break
            self.persistence.write_queue(queue)

    def update_phase(self, task_id: str, phase: str) -> None:
        """Update current workflow phase for checkpointing.

        Args:
            task_id: Task UUID
            phase: Current phase name
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["current_phase"] = phase
                    break
            self.persistence.write_queue(queue)

    def list_tasks(
        self,
        status: Optional[TaskStatus | str] = None,
        category: Optional[TaskCategory | str] = None,
    ) -> list[Task]:
        """List tasks with optional filtering.

        Args:
            status: Filter by status
            category: Filter by category

        Returns:
            List of matching tasks
        """
        # Normalize to values
        status_val = status.value if isinstance(status, TaskStatus) else status
        cat_val = category.value if isinstance(category, TaskCategory) else category

        with self.persistence.lock():
            queue = self.persistence.read_queue()
            tasks = queue["topics"]

            if status_val:
                tasks = [t for t in tasks if t["status"] == status_val]
            if cat_val:
                tasks = [t for t in tasks if t["category"] == cat_val]

            return tasks

    def reorder(self, task_ids: list[str]) -> None:
        """Reorder tasks by providing list of task IDs in desired order.

        Useful for LLM-based queue editing. Tasks not in the list
        are appended at the end.

        Args:
            task_ids: Task IDs in desired order
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()

            # Build lookup
            task_map = {t["id"]: t for t in queue["topics"]}

            # Reorder
            reordered = [task_map[tid] for tid in task_ids if tid in task_map]

            # Add any tasks not in the list (safety)
            seen = set(task_ids)
            for task in queue["topics"]:
                if task["id"] not in seen:
                    reordered.append(task)

            queue["topics"] = reordered
            self.persistence.write_queue(queue)

    def set_concurrency(
        self,
        mode: str,
        max_concurrent: int = 1,
        stagger_hours: float = 36.0,
    ) -> None:
        """Update concurrency configuration.

        Args:
            mode: "max_concurrent" or "stagger_hours"
            max_concurrent: Max simultaneous tasks
            stagger_hours: Hours between task starts
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            queue["concurrency"] = {
                "mode": mode,
                "max_concurrent": max_concurrent,
                "stagger_hours": stagger_hours,
            }
            self.persistence.write_queue(queue)

    def get_concurrency_config(self) -> ConcurrencyConfig:
        """Get current concurrency configuration."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            return queue["concurrency"]

    def get_categories(self) -> list[str]:
        """Get list of categories."""
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            return queue["categories"]

    def set_categories(self, categories: list[str]) -> None:
        """Update category list.

        Args:
            categories: New category list
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            queue["categories"] = categories
            # Reset rotation if categories changed
            queue["last_category_index"] = -1
            self.persistence.write_queue(queue)

    def get_queue_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dict with counts by status and category
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            tasks = queue["topics"]

            by_status = {}
            by_category = {}

            for task in tasks:
                # Count by status
                status = task["status"]
                by_status[status] = by_status.get(status, 0) + 1

                # Count by category
                category = task["category"]
                by_category[category] = by_category.get(category, 0) + 1

            return {
                "total": len(tasks),
                "by_status": by_status,
                "by_category": by_category,
                "concurrency": queue["concurrency"],
            }

    def find_next_available_monday(
        self,
        category: str,
        timezone: str = "Pacific/Auckland",
    ) -> datetime:
        """Find next Monday 3pm that doesn't conflict with existing publish_series.

        Scans existing publish_series tasks for the same category and finds
        the first Monday that isn't already scheduled.

        Args:
            category: Task category for conflict checking
            timezone: Timezone for local time calculations

        Returns:
            datetime at next available Monday 3pm local time
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            return self.publishing.find_next_available_monday(queue, category, timezone)

    def update_task(self, task_id: str, **updates) -> bool:
        """Update arbitrary fields on a task.

        Args:
            task_id: Task UUID
            **updates: Fields to update

        Returns:
            True if task was found and updated
        """
        with self.persistence.lock():
            queue = self.persistence.read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task.update(updates)
                    self.persistence.write_queue(queue)
                    return True
        return False
