"""Queue persistence with file locking and atomic writes."""

import fcntl
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import TaskQueue

logger = logging.getLogger(__name__)


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
        """Read queue from disk."""
        with open(self.queue_file, "r") as f:
            return json.load(f)

    def write_queue(self, queue: TaskQueue) -> None:
        """Write queue to disk atomically."""
        queue["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Write to temp file first, then rename for atomicity
        temp_file = self.queue_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(queue, f, indent=2)
        try:
            temp_file.rename(self.queue_file)
        except FileNotFoundError:
            # Temp file may have been deleted by concurrent cleanup
            logger.warning(f"Temp file {temp_file} disappeared before rename - retrying write")
            with open(self.queue_file, "w") as f:
                json.dump(queue, f, indent=2)
