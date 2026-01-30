"""Concurrency validation for task queue."""

from datetime import datetime, timezone

from .schemas import TaskQueue, TaskStatus


class ConcurrencyValidator:
    """Validates whether a new task can start based on concurrency config."""

    @staticmethod
    def can_start_new_task(queue: TaskQueue) -> bool:
        """Check if concurrency constraints allow starting a new task.

        Args:
            queue: Current task queue

        Returns:
            True if a new task can start
        """
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
