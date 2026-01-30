"""Publishing date calculation for task queue."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .schemas import TaskQueue, TaskStatus


class PublishingScheduler:
    """Finds available publishing dates avoiding conflicts."""

    @staticmethod
    def find_next_available_monday(
        queue: TaskQueue,
        category: str,
        timezone_str: str = "Pacific/Auckland",
    ) -> datetime:
        """Find next Monday 3pm that doesn't conflict with existing publish_series.

        Scans existing publish_series tasks for the same category and finds
        the first Monday that isn't already scheduled.

        Args:
            queue: Current task queue
            category: Task category for conflict checking
            timezone_str: Timezone for local time calculations

        Returns:
            datetime at next available Monday 3pm local time
        """
        local_tz = ZoneInfo(timezone_str)

        # Start from next Monday
        now = datetime.now(local_tz)
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7  # Next Monday, not today

        candidate = now.replace(hour=15, minute=0, second=0, microsecond=0)
        candidate += timedelta(days=days_until_monday)

        # Get existing publish_series tasks for this category
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
