"""Save-and-spawn phase: save unillustrated articles, publish lit review, spawn illustrate_and_publish task."""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def run_save_and_spawn_phase(
    task: dict[str, Any],
    final_report: str,
    final_outputs: list[dict],
    get_output_dir_fn,
    slugify_fn,
) -> dict[str, Any]:
    """Save unillustrated articles to disk, publish lit review as draft, spawn illustrate_and_publish task.

    Args:
        task: Task data
        final_report: Enhanced literature review markdown
        final_outputs: Evening reads article outputs
        get_output_dir_fn: Function to get output directory
        slugify_fn: Function to slugify strings

    Returns:
        Dict with illustrate_task_id and lit_review_draft_url
    """
    task_id = task["id"]
    topic = task.get("topic", "unknown")
    category = task.get("category", "")
    quality = task.get("quality", "standard")
    topic_slug = slugify_fn(topic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        articles.append({
            "id": article_id,
            "title": output["title"],
            "filename": filename,
        })

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
    _write_atomic(manifest_path, manifest)
    logger.info(f"Wrote manifest with {len(articles)} articles")

    # Publish lit review as Substack draft (audience: only_paid)
    lit_review_draft_url = None
    try:
        lit_review_draft_url = await _publish_lit_review_draft(
            final_report, topic, category
        )
    except Exception as e:
        logger.error(f"Failed to publish lit review draft: {e}")

    # Spawn illustrate_and_publish task
    illustrate_task_id = _spawn_illustrate_task(
        task=task,
        manifest_path=str(manifest_path),
        unillust_dir=unillust_dir,
        articles=articles,
        final_outputs=final_outputs,
    )

    return {
        "illustrate_task_id": illustrate_task_id,
        "lit_review_draft_url": lit_review_draft_url,
        "output_dir": str(unillust_dir),
    }


def _write_atomic(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + rename."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp_path).rename(path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise


async def _publish_lit_review_draft(
    content: str, topic: str, category: str
) -> str | None:
    """Publish the lit review as a Substack draft. Returns draft URL."""
    from core.task_queue.paths import PUBLICATIONS_FILE, SUBSTACK_COOKIES_FILE
    from utils.substack_publish import SubstackConfig, SubstackPublisher

    if not PUBLICATIONS_FILE.exists():
        logger.warning("publications.json not found, skipping lit review draft")
        return None

    with open(PUBLICATIONS_FILE) as f:
        pubs = json.load(f)

    pub_config = pubs.get(category) or (next(iter(pubs.values())) if pubs else {})

    config = SubstackConfig(
        cookies_path=str(SUBSTACK_COOKIES_FILE),
        publication_url=pub_config.get("publication_url"),
        audience="only_paid",
    )
    publisher = SubstackPublisher(config)

    result = await asyncio.to_thread(
        publisher.create_draft,
        markdown=content,
        title=f"Research Deep-Dive: {topic}",
    )

    if result.get("success"):
        logger.info(f"Published lit review draft: {result.get('draft_url')}")
        return result.get("draft_url")
    else:
        logger.error(f"Lit review draft failed: {result.get('error')}")
        return None


def _spawn_illustrate_task(
    task: dict[str, Any],
    manifest_path: str,
    unillust_dir: Path,
    articles: list[dict],
    final_outputs: list[dict],
) -> str:
    """Create an illustrate_and_publish task in the queue."""
    from core.task_queue.queue_manager import TaskQueueManager

    # Build title lookup
    titles = {out["id"]: out["title"] for out in final_outputs}

    # Build items list
    items = []
    for article in articles:
        items.append({
            "id": article["id"],
            "title": titles.get(article["id"], article["id"]),
            "source_path": str(unillust_dir / article["filename"]),
            "illustrated": False,
            "illustrated_path": None,
            "draft_id": None,
            "draft_url": None,
        })

    queue = TaskQueueManager()
    return queue.add_task(
        task_type="illustrate_and_publish",
        category=task["category"],
        priority=task["priority"],
        quality=task.get("quality", "standard"),
        topic=task.get("topic", ""),
        source_task_id=task["id"],
        manifest_path=manifest_path,
        items=items,
    )
