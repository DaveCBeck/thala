"""Queue persistence with file locking and atomic writes."""

import fcntl
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import TaskQueue

logger = logging.getLogger(__name__)

# Task types that belong in the publish queue
PUBLISH_TASK_TYPES = {"illustrate_and_export"}


class QueuePersistence:
    """Handles file locking, reading, and writing for queue.json."""

    def __init__(self, queue_file: Path, lock_file: Path):
        """Initialize persistence handler.

        Args:
            queue_file: Path to queue.json
            lock_file: Path to lock file
        """
        self.queue_file = queue_file
        self.lock_file = lock_file

    @contextmanager
    def lock(self):
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

    def read_queue(self) -> TaskQueue:
        """Read queue from disk, auto-migrating v1 format if needed."""
        with open(self.queue_file, "r") as f:
            data = json.load(f)

        # Auto-migrate v1 → v2
        if "topics" in data:
            data = self._migrate_v1_to_v2(data)
            self.write_queue(data)
            logger.info("Migrated queue.json from v1 to v2 format")

        return data

    def write_queue(self, queue: TaskQueue) -> None:
        """Write queue to disk atomically."""
        queue["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Write to temp file first, then rename for atomicity
        temp_file = self.queue_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(queue, f, indent=2)
        temp_file.rename(self.queue_file)

    @staticmethod
    def _migrate_v1_to_v2(data: dict) -> TaskQueue:
        """Migrate v1 queue format (single topics list) to v2 (two arrays).

        Partitions tasks by task_type into research_tasks and publish_tasks.
        Drops the concurrency config block (no longer used).
        """
        research_tasks = []
        publish_tasks = []

        for task in data.get("topics", []):
            task_type = task.get("task_type", "lit_review_full")
            if task_type in PUBLISH_TASK_TYPES:
                publish_tasks.append(task)
            else:
                research_tasks.append(task)

        return {
            "version": "2.0",
            "categories": data.get("categories", []),
            "last_category_index": data.get("last_category_index", -1),
            "research_tasks": research_tasks,
            "publish_tasks": publish_tasks,
            "last_updated": data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        }
