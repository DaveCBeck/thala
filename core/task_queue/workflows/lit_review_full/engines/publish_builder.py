"""Publish task builder."""

from typing import Any, Optional


def spawn_publish_task(
    task: dict[str, Any],
    lit_review_path: Optional[str],
    illustrated_paths: dict[str, str],
    final_outputs: list[dict],
) -> str:
    """Create a publish_series task with schedule.

    Args:
        task: Parent task data
        lit_review_path: Path to illustrated lit review
        illustrated_paths: Map of article ID to illustrated file path
        final_outputs: Evening reads outputs (for titles)

    Returns:
        New task ID
    """
    from core.task_queue.queue_manager import TaskQueueManager

    queue = TaskQueueManager()
    base_date = queue.find_next_available_monday(task["category"])

    # Build title lookup from final_outputs
    titles = {out["id"]: out["title"] for out in final_outputs}

    # Build publish items following the schedule:
    # Day 0: Overview (everyone)
    # Day 1: Lit Review (only_paid)
    # Day 4: Deep Dive 1 (everyone)
    # Day 7: Deep Dive 2 (everyone)
    # Day 11: Deep Dive 3 (everyone)
    items = [
        {
            "id": "overview",
            "title": titles.get("overview", f"Overview: {task['topic']}"),
            "path": illustrated_paths.get("overview", ""),
            "day_offset": 0,
            "audience": "everyone",
            "published": False,
            "draft_id": None,
            "draft_url": None,
        },
        {
            "id": "lit_review",
            "title": f"Research Deep-Dive: {task['topic']}",
            "path": lit_review_path or "",
            "day_offset": 1,
            "audience": "only_paid",
            "published": False,
            "draft_id": None,
            "draft_url": None,
        },
        {
            "id": "deep_dive_1",
            "title": titles.get("deep_dive_1", f"Deep Dive 1: {task['topic']}"),
            "path": illustrated_paths.get("deep_dive_1", ""),
            "day_offset": 4,
            "audience": "everyone",
            "published": False,
            "draft_id": None,
            "draft_url": None,
        },
        {
            "id": "deep_dive_2",
            "title": titles.get("deep_dive_2", f"Deep Dive 2: {task['topic']}"),
            "path": illustrated_paths.get("deep_dive_2", ""),
            "day_offset": 7,
            "audience": "everyone",
            "published": False,
            "draft_id": None,
            "draft_url": None,
        },
        {
            "id": "deep_dive_3",
            "title": titles.get("deep_dive_3", f"Deep Dive 3: {task['topic']}"),
            "path": illustrated_paths.get("deep_dive_3", ""),
            "day_offset": 11,
            "audience": "everyone",
            "published": False,
            "draft_id": None,
            "draft_url": None,
        },
    ]

    return queue.add_task(
        task_type="publish_series",
        category=task["category"],
        priority=task["priority"],
        quality=task.get("quality", "standard"),
        base_date=base_date.isoformat(),
        items=items,
        source_task_id=task["id"],
    )
