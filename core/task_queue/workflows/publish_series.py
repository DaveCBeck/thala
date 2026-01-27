"""Publish series workflow.

Schedule-aware draft publishing via Substack API.
Spawned by lit_review_full after illustration completes.

This workflow:
1. Checks which items are due for publishing based on base_date + day_offset
2. Creates drafts on Substack for due items
3. Updates task items with draft IDs and URLs
4. Completes when all items have been published
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from ..paths import PUBLICATIONS_FILE, SUBSTACK_COOKIES_FILE
from .base import BaseWorkflow

logger = logging.getLogger(__name__)

# Publication config location
PUBLICATIONS_CONFIG = PUBLICATIONS_FILE


class PublishSeriesWorkflow(BaseWorkflow):
    """Schedule-aware draft publishing workflow."""

    @property
    def task_type(self) -> str:
        return "publish_series"

    @property
    def phases(self) -> list[str]:
        return ["checking", "publishing", "complete"]

    @property
    def is_zero_cost(self) -> bool:
        """This workflow makes no LLM calls, skip budget check."""
        return True

    def get_task_identifier(self, task: dict[str, Any]) -> str:
        """Get identifier for logging."""
        source_id = task.get("source_task_id", "unknown")[:8]
        return f"publish_series({source_id})"

    async def run(
        self,
        task: dict[str, Any],
        checkpoint_callback: Callable[[str], None],
        resume_from: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Run the publish series workflow.

        Checks which items are due and publishes them as drafts.

        Args:
            task: PublishSeriesTask with base_date, items, etc.
            checkpoint_callback: Progress callback
            resume_from: Optional checkpoint for resumption

        Returns:
            Dict with status, published items, etc.
        """
        checkpoint_callback("checking")

        base_date = datetime.fromisoformat(task["base_date"])
        now = datetime.now()
        items = task["items"]

        # Find items due for publishing
        items_to_publish = []
        for item in items:
            if item["published"]:
                continue
            publish_date = base_date + timedelta(days=item["day_offset"])
            if now >= publish_date:
                items_to_publish.append(item)

        if not items_to_publish:
            # Nothing due yet - find next publish date
            next_date = self._get_next_publish_date(task)
            logger.info(f"No items due yet. Next publish: {next_date}")
            return {
                "status": "waiting",
                "next_publish": next_date,
                "published_this_run": [],
                "all_complete": False,
            }

        checkpoint_callback("publishing")

        # Load publication config
        pub_config = self._load_publication_config(task["category"])

        # Import here to avoid circular imports
        from utils.substack_publish import SubstackPublisher, SubstackConfig

        published_items = []
        errors = []

        for item in items_to_publish:
            try:
                # Read markdown content
                content_path = Path(item["path"])
                if not content_path.exists():
                    logger.error(f"Content file not found: {item['path']}")
                    errors.append({"item": item["id"], "error": "File not found"})
                    continue

                content = content_path.read_text()

                # Create publisher with per-item audience
                item_config = SubstackConfig(
                    cookies_path=str(SUBSTACK_COOKIES_FILE),
                    publication_url=pub_config.get("publication_url"),
                    audience=item.get("audience", "everyone"),
                )
                publisher = SubstackPublisher(item_config)

                # Create draft
                result = publisher.create_draft(
                    markdown=content,
                    title=item["title"],
                )

                if result["success"]:
                    item["published"] = True
                    item["draft_id"] = result["post_id"]
                    item["draft_url"] = result["draft_url"]
                    published_items.append(item["id"])
                    logger.info(f"Created draft: {item['title']} -> {result['draft_url']}")
                else:
                    errors.append({"item": item["id"], "error": result.get("error", "Unknown error")})
                    logger.error(f"Failed to create draft for {item['id']}: {result.get('error')}")

            except Exception as e:
                logger.exception(f"Error publishing {item['id']}")
                errors.append({"item": item["id"], "error": str(e)})

        # Update task items in queue
        self._update_task_items(task["id"], items)

        # Check if all done
        all_published = all(item["published"] for item in items)

        if all_published:
            status = "success"
        elif published_items:
            status = "partial"
        elif errors:
            status = "failed"
        else:
            status = "waiting"

        return {
            "status": status,
            "published_this_run": published_items,
            "all_complete": all_published,
            "errors": errors if errors else None,
        }

    def save_outputs(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, str]:
        """Save publish results summary.

        Args:
            task: Task data
            result: Workflow result

        Returns:
            Dict with paths to saved files (empty for this workflow)
        """
        # No files to save - updates are in the task itself
        return {}

    def _load_publication_config(self, category: str) -> dict:
        """Load publication config for a category.

        Args:
            category: Task category (e.g., "technology")

        Returns:
            Dict with publication_url, subdomain, etc.
        """
        if not PUBLICATIONS_CONFIG.exists():
            logger.warning(f"Publications config not found: {PUBLICATIONS_CONFIG}")
            return {}

        with open(PUBLICATIONS_CONFIG) as f:
            pubs = json.load(f)

        # Return config for category, or first available
        if category in pubs:
            return pubs[category]

        # Fallback to first category
        if pubs:
            return next(iter(pubs.values()))

        return {}

    def _get_next_publish_date(self, task: dict[str, Any]) -> Optional[str]:
        """Get the next publish date for unpublished items.

        Args:
            task: PublishSeriesTask

        Returns:
            ISO datetime string of next publish, or None if all published
        """
        base_date = datetime.fromisoformat(task["base_date"])
        items = task["items"]

        unpublished = [
            item for item in items
            if not item["published"]
        ]

        if not unpublished:
            return None

        # Find earliest unpublished item
        next_item = min(unpublished, key=lambda x: x["day_offset"])
        next_date = base_date + timedelta(days=next_item["day_offset"])

        return next_date.isoformat()

    def _update_task_items(self, task_id: str, items: list[dict]) -> None:
        """Update task items in queue.

        Args:
            task_id: Task UUID
            items: Updated items list
        """
        from ..queue_manager import TaskQueueManager

        queue = TaskQueueManager()
        with queue._lock():
            queue_data = queue._read_queue()
            for task in queue_data["topics"]:
                if task["id"] == task_id:
                    task["items"] = items
                    break
            queue._write_queue(queue_data)
