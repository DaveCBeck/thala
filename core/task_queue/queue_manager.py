"""
Task queue manager with safe concurrent access.

Provides:
- File locking via fcntl for cross-process coordination
- Atomic writes via temp file + rename
- Round-robin category selection with priority within category
- Flexible concurrency control (max_concurrent or stagger_hours)
"""

import fcntl
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from .paths import PUBLICATIONS_FILE, QUEUE_DIR
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

# Fallback categories if publications.json doesn't exist
_FALLBACK_CATEGORIES = [
    "philosophy",
    "science",
    "technology",
    "society",
    "culture",
]


def load_categories_from_publications(publications_file: Path = PUBLICATIONS_FILE) -> list[str]:
    """Load category names from publications.json.

    publications.json is the source of truth for categories.
    Categories are derived from the top-level keys in the file.

    Args:
        publications_file: Path to publications.json

    Returns:
        List of category names
    """
    if publications_file.exists():
        try:
            with open(publications_file) as f:
                pubs = json.load(f)
            return list(pubs.keys())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load publications.json: {e}")

    return _FALLBACK_CATEGORIES


# Load categories from publications.json (source of truth)
DEFAULT_CATEGORIES = load_categories_from_publications()


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

        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_queue_exists()

    def _ensure_queue_exists(self) -> None:
        """Create queue file if it doesn't exist."""
        if not self.queue_file.exists():
            initial_queue: TaskQueue = {
                "version": "1.0",
                "concurrency": {
                    "mode": "stagger_hours",
                    "max_concurrent": 1,
                    "stagger_hours": 36.0,
                },
                "categories": DEFAULT_CATEGORIES,
                "last_category_index": -1,
                "topics": [],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self._write_queue(initial_queue)

    @contextmanager
    def _lock(self):
        """Acquire exclusive lock on queue file.

        Uses fcntl.flock for cross-process coordination.
        """
        self.lock_file.touch(exist_ok=True)
        lock_fd = open(self.lock_file, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()

    def _read_queue(self) -> TaskQueue:
        """Read queue from disk."""
        with open(self.queue_file, "r") as f:
            return json.load(f)

    def _write_queue(self, queue: TaskQueue) -> None:
        """Write queue to disk atomically."""
        queue["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Write to temp file first, then rename for atomicity
        temp_file = self.queue_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(queue, f, indent=2)
        temp_file.rename(self.queue_file)

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

        with self._lock():
            queue = self._read_queue()
            queue["topics"].append(new_task)
            self._write_queue(queue)

        logger.info(f"Added task {task_id} ({task_type}): {identifier[:50]}...")
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: Task UUID

        Returns:
            Task if found, None otherwise
        """
        with self._lock():
            queue = self._read_queue()
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
        with self._lock():
            queue = self._read_queue()

            # Check concurrency constraints
            if not self._can_start_new_task(queue):
                return None

            # Get pending tasks
            pending = [
                t for t in queue["topics"] if t["status"] == TaskStatus.PENDING.value
            ]

            if not pending:
                return None

            # Load categories fresh from publications.json (source of truth)
            categories = load_categories_from_publications(
                self.queue_dir / "publications.json"
            )
            last_idx = queue.get("last_category_index", -1)

            # Sync queue categories if they've changed
            if queue.get("categories") != categories:
                queue["categories"] = categories
                # Reset index if it's out of bounds
                if last_idx >= len(categories):
                    last_idx = -1
                    queue["last_category_index"] = last_idx

            # Try each category in round-robin order
            for offset in range(len(categories)):
                cat_idx = (last_idx + 1 + offset) % len(categories)
                category = categories[cat_idx]

                # Find highest priority task in this category
                cat_tasks = [t for t in pending if t["category"] == category]

                if cat_tasks:
                    # Sort by priority (descending), then created_at (ascending)
                    cat_tasks.sort(key=lambda t: (-t["priority"], t["created_at"]))
                    selected = cat_tasks[0]

                    # Update category index
                    queue["last_category_index"] = cat_idx
                    self._write_queue(queue)

                    return selected

            # If no tasks in current categories, fall back to highest priority overall
            # This handles tasks with categories that were removed from publications.json
            pending.sort(key=lambda t: (-t["priority"], t["created_at"]))
            logger.info(
                f"No tasks in current categories, falling back to task with "
                f"category '{pending[0]['category']}' (may be deprecated)"
            )
            return pending[0]

    def _can_start_new_task(self, queue: TaskQueue) -> bool:
        """Check if concurrency constraints allow starting a new task."""
        config = queue["concurrency"]
        in_progress = [
            t for t in queue["topics"] if t["status"] == TaskStatus.IN_PROGRESS.value
        ]

        if config["mode"] == "max_concurrent":
            return len(in_progress) < config["max_concurrent"]

        elif config["mode"] == "stagger_hours":
            if not in_progress:
                return True

            # Find most recently started task
            started_times = [
                datetime.fromisoformat(t["started_at"])
                for t in in_progress
                if t["started_at"]
            ]

            if not started_times:
                return True

            latest_start = max(started_times)
            hours_elapsed = (datetime.now(timezone.utc) - latest_start).total_seconds() / 3600

            return hours_elapsed >= config["stagger_hours"]

        return True

    def mark_started(self, task_id: str, langsmith_run_id: str) -> None:
        """Mark task as started.

        Args:
            task_id: Task UUID
            langsmith_run_id: LangSmith run ID for cost tracking
        """
        with self._lock():
            queue = self._read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["status"] = TaskStatus.IN_PROGRESS.value
                    task["started_at"] = datetime.now(timezone.utc).isoformat()
                    task["langsmith_run_id"] = langsmith_run_id
                    break
            self._write_queue(queue)

    def mark_completed(self, task_id: str) -> None:
        """Mark task as completed."""
        with self._lock():
            queue = self._read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["status"] = TaskStatus.COMPLETED.value
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    break
            self._write_queue(queue)

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed.

        Args:
            task_id: Task UUID
            error: Error message
        """
        with self._lock():
            queue = self._read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["status"] = TaskStatus.FAILED.value
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    task["error_message"] = error
                    break
            self._write_queue(queue)

    def update_phase(self, task_id: str, phase: str) -> None:
        """Update current workflow phase for checkpointing.

        Args:
            task_id: Task UUID
            phase: Current phase name
        """
        with self._lock():
            queue = self._read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task["current_phase"] = phase
                    break
            self._write_queue(queue)

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

        with self._lock():
            queue = self._read_queue()
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
        with self._lock():
            queue = self._read_queue()

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
            self._write_queue(queue)

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
        with self._lock():
            queue = self._read_queue()
            queue["concurrency"] = {
                "mode": mode,
                "max_concurrent": max_concurrent,
                "stagger_hours": stagger_hours,
            }
            self._write_queue(queue)

    def get_concurrency_config(self) -> ConcurrencyConfig:
        """Get current concurrency configuration."""
        with self._lock():
            queue = self._read_queue()
            return queue["concurrency"]

    def get_categories(self) -> list[str]:
        """Get list of categories."""
        with self._lock():
            queue = self._read_queue()
            return queue["categories"]

    def set_categories(self, categories: list[str]) -> None:
        """Update category list.

        Args:
            categories: New category list
        """
        with self._lock():
            queue = self._read_queue()
            queue["categories"] = categories
            # Reset rotation if categories changed
            queue["last_category_index"] = -1
            self._write_queue(queue)

    def get_queue_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dict with counts by status and category
        """
        with self._lock():
            queue = self._read_queue()
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
        local_tz = ZoneInfo(timezone)

        # Start from next Monday
        now = datetime.now(local_tz)
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7  # Next Monday, not today

        candidate = now.replace(hour=15, minute=0, second=0, microsecond=0)
        candidate += timedelta(days=days_until_monday)

        # Get existing publish_series tasks for this category
        with self._lock():
            queue = self._read_queue()
            publish_tasks = [
                t for t in queue["topics"]
                if t.get("task_type") == "publish_series"
                and t.get("category") == category
                and t.get("status") != TaskStatus.COMPLETED.value
            ]

        # Extract existing base_dates
        existing_dates = set()
        for t in publish_tasks:
            base_date_str = t.get("base_date")
            if base_date_str:
                try:
                    bd = datetime.fromisoformat(base_date_str)
                    existing_dates.add(bd.date())
                except ValueError:
                    pass

        # Find first available Monday
        max_attempts = 52  # Don't look more than a year ahead
        for _ in range(max_attempts):
            if candidate.date() not in existing_dates:
                return candidate
            candidate += timedelta(days=7)

        # Fallback (shouldn't happen)
        return candidate

    def update_task(self, task_id: str, **updates) -> bool:
        """Update arbitrary fields on a task.

        Args:
            task_id: Task UUID
            **updates: Fields to update

        Returns:
            True if task was found and updated
        """
        with self._lock():
            queue = self._read_queue()
            for task in queue["topics"]:
                if task["id"] == task_id:
                    task.update(updates)
                    self._write_queue(queue)
                    return True
        return False
