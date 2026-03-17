"""Save-and-spawn phase: save unillustrated articles, export lit review to Quartz, spawn illustrate_and_export task."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.task_queue.utils import write_json_atomic

logger = logging.getLogger(__name__)


async def run_save_and_spawn_phase(
    task: dict[str, Any],
    final_report: str,
    final_outputs: list[dict],
    get_output_dir_fn: Callable[[], Path],
    slugify_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Save unillustrated articles to disk, export lit review to Quartz, spawn illustrate_and_export task.

    Args:
        task: Task data
        final_report: Enhanced literature review markdown
        final_outputs: Evening reads article outputs
        get_output_dir_fn: Function to get output directory
        slugify_fn: Function to slugify strings

    Returns:
        Dict with illustrate_task_id, quartz_path, output_dir
    """
    task_id = task["id"]
    topic = task.get("topic", "unknown")
    category = task.get("category", "")
    quality = task.get("quality", "standard")
    topic_slug = slugify_fn(topic)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Create output directory with task ID to prevent collisions
    output_dir = get_output_dir_fn()
    unillust_dir = output_dir / f"unillustrated_{topic_slug}_{task_id[:8]}_{timestamp}"
    unillust_dir.mkdir(exist_ok=True)

    # Save lit review
    lit_review_path = unillust_dir / "lit_review.md"
    lit_review_path.write_text(final_report)
    logger.info(f"Saved lit review to {lit_review_path}")

    # Save each evening reads article
    articles = []
    for output in final_outputs:
        article_id = output["id"]
        filename = f"{article_id}.md"
        article_path = unillust_dir / filename
        article_path.write_text(output["content"])
        articles.append(
            {
                "id": article_id,
                "title": output["title"],
                "subtitle": output.get("subtitle", ""),
                "filename": filename,
            }
        )

    # Write manifest atomically (write to tmp, then rename) — acts as commit point
    manifest = {
        "topic": topic,
        "category": category,
        "quality": quality,
        "source_task_id": task_id,
        "output_dir": str(unillust_dir),
        "articles": articles,
    }
    manifest_path = unillust_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest, indent=2)
    logger.info(f"Wrote manifest with {len(articles)} articles")

    # Export full review to Quartz site
    quartz_path = None
    try:
        from core.task_queue.workflows.shared.quartz_export import export_lit_review_to_quartz

        quartz_path = await export_lit_review_to_quartz(
            content=final_report,
            topic=topic,
            category=category,
            generated_at=datetime.now(timezone.utc).isoformat(),
            quality=quality,
        )
    except Exception as e:
        logger.error(f"Failed to export lit review to Quartz: {e}")

    # Spawn illustrate_and_export task
    illustrate_task_id = await _spawn_illustrate_task(
        task=task,
        manifest_path=str(manifest_path),
        unillust_dir=unillust_dir,
        articles=articles,
        final_outputs=final_outputs,
    )

    return {
        "illustrate_task_id": illustrate_task_id,
        "quartz_path": str(quartz_path) if quartz_path else None,
        "output_dir": str(unillust_dir),
    }


async def _spawn_illustrate_task(
    task: dict[str, Any],
    manifest_path: str,
    unillust_dir: Path,
    articles: list[dict],
    final_outputs: list[dict],
) -> str:
    """Create an illustrate_and_export task in the queue."""
    from core.task_queue.queue_manager import TaskQueueManager

    # Build lookups from final outputs
    titles = {out["id"]: out["title"] for out in final_outputs}
    subtitles = {out["id"]: out.get("subtitle", "") for out in final_outputs}

    # Build items list
    items = []
    for article in articles:
        items.append(
            {
                "id": article["id"],
                "title": titles.get(article["id"], article["id"]),
                "subtitle": subtitles.get(article["id"], ""),
                "source_path": str(unillust_dir / article["filename"]),
                "illustrated": False,
                "illustrated_path": None,
                "exported": False,
            }
        )

    queue = TaskQueueManager()
    return await asyncio.to_thread(
        queue.add_task,
        task_type="illustrate_and_export",
        category=task["category"],
        priority=task["priority"],
        quality=task.get("quality", "standard"),
        topic=task.get("topic", ""),
        source_task_id=task["id"],
        manifest_path=manifest_path,
        items=items,
        not_before=None,  # Immediately eligible
    )
