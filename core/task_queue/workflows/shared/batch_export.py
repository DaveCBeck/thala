"""Batch export: write publish-ready articles to a staging directory for rsync to VPS.

Each batch folder contains illustrated markdown (with relative image paths),
image directories, the lit review, and a manifest.json describing all articles.
"""

import logging
import re
import shutil

from datetime import datetime, timezone
from pathlib import Path

from core.task_queue.paths import EXPORT_DIR
from core.task_queue.pub_counter import next_id
from core.task_queue.utils import write_json_atomic
from core.task_queue.workflows.shared.publication_config import load_publication_config

logger = logging.getLogger(__name__)


def rewrite_image_paths(content: str, abs_output_dir: str) -> str:
    """Rewrite absolute image paths to relative paths.

    Replaces e.g. ![alt](/home/.../overview_images/header.png)
    with        ![alt](./overview_images/header.png)
    """
    # Normalise to ensure trailing slash for clean replacement
    prefix = abs_output_dir.rstrip("/") + "/"
    # Match markdown image syntax with this absolute prefix
    pattern = re.compile(
        r"(!\[[^\]]*\]\()" + re.escape(prefix) + r"([^)]+\))"
    )
    return pattern.sub(r"\1./\2", content)


def export_batch(
    output_dir: Path,
    manifest: dict,
    items: list[dict],
    category: str,
) -> Path:
    """Export illustrated articles to a batch folder for VPS transfer.

    Args:
        output_dir: The unillustrated_* directory containing source files.
        manifest: The original manifest.json data from lit_review_full.
        items: The workflow items list (with illustrated_path, title, etc.).
        category: Task category for publication lookup.

    Returns:
        Path to the created batch directory.
    """
    pub_config = load_publication_config(category)
    pub_slug = pub_config.get("subdomain", "unknown")
    pub_url = pub_config.get("publication_url", "")

    seq_id = next_id(pub_slug)
    batch_name = f"batch_{seq_id:04d}"
    batch_dir = EXPORT_DIR / pub_slug / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    abs_output_dir = str(output_dir)
    topic = manifest.get("topic", "")
    batch_articles = []

    # Export illustrated articles
    for item in items:
        illustrated_path = item.get("illustrated_path")
        if not illustrated_path:
            logger.warning("Skipping %s — no illustrated_path", item["id"])
            continue

        src = Path(illustrated_path)
        if not src.exists():
            logger.warning("Illustrated file missing: %s", src)
            continue

        # Copy and rewrite markdown
        content = src.read_text()
        content = rewrite_image_paths(content, abs_output_dir)
        dest = batch_dir / src.name
        dest.write_text(content)

        # Copy images directory
        images_dir_name = f"{item['id']}_images"
        images_src = output_dir / images_dir_name
        images_dest = batch_dir / images_dir_name
        if images_src.is_dir():
            shutil.copytree(images_src, images_dest, dirs_exist_ok=True)

        article_type = "overview" if item["id"] == "overview" else "deep_dive"
        batch_articles.append({
            "id": item["id"],
            "title": item.get("title", item["id"]),
            "draft_subtitle": item.get("subtitle", ""),
            "filename": src.name,
            "images_dir": images_dir_name if images_src.is_dir() else None,
            "audience": "everyone",
            "type": article_type,
            "pub_seq_id": seq_id,
            "status": "pending",
        })

    # Copy lit review (no images, not illustrated)
    lit_review_src = output_dir / "lit_review.md"
    if lit_review_src.exists():
        shutil.copy2(lit_review_src, batch_dir / "lit_review.md")
        batch_articles.append({
            "id": "lit_review",
            "title": f"Literature Review: {topic}",
            "filename": "lit_review.md",
            "images_dir": None,
            "audience": "only_paid",
            "type": "lit_review",
            "pub_seq_id": seq_id,
            "status": "pending",
        })

    # Write batch manifest
    batch_manifest = {
        "batch_id": batch_name,
        "topic": topic,
        "category": category,
        "publication_slug": pub_slug,
        "publication_url": pub_url,
        "source_task_id": manifest.get("source_task_id", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "articles": batch_articles,
    }
    write_json_atomic(batch_dir / "manifest.json", batch_manifest, indent=2)

    logger.info("Exported batch %s/%s with %d articles", pub_slug, batch_name, len(batch_articles))
    return batch_dir
