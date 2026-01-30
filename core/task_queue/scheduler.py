"""Task scheduling with round-robin category selection."""

import logging
from pathlib import Path
from typing import Optional

from .categories import load_categories_from_publications
from .concurrency import ConcurrencyValidator
from .schemas import Task, TaskQueue, TaskStatus

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Selects next eligible task using round-robin category selection."""

    def __init__(self, publications_file: Path):
        """Initialize scheduler.

        Args:
            publications_file: Path to publications.json for loading categories
        """
        self.publications_file = publications_file
        self.validator = ConcurrencyValidator()

    def get_next_eligible_task(self, queue: TaskQueue) -> Optional[Task]:
        """Get next task eligible to run.

        Selection considers:
        1. Concurrency limits (max_concurrent or stagger_hours)
        2. Category rotation (round-robin)
        3. Priority (within same category rotation)

        Args:
            queue: Current task queue

        Returns:
            Next eligible task, or None if none eligible
        """
        # Check concurrency constraints
        if not self.validator.can_start_new_task(queue):
            return None

        # Get pending tasks
        pending = [
            t for t in queue["topics"] if t["status"] == TaskStatus.PENDING.value
        ]

        if not pending:
            return None

        # Load categories fresh from publications.json (source of truth)
        categories = load_categories_from_publications(self.publications_file)
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

                return selected

        # If no tasks in current categories, fall back to highest priority overall
        # This handles tasks with categories that were removed from publications.json
        pending.sort(key=lambda t: (-t["priority"], t["created_at"]))
        logger.info(
            f"No tasks in current categories, falling back to task with "
            f"category '{pending[0]['category']}' (may be deprecated)"
        )
        return pending[0]
