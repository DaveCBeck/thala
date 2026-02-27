"""Export literature reviews to the Quartz site content directory."""

import logging
import re
from pathlib import Path

from core.task_queue.paths import QUARTZ_CONTENT_DIR

logger = logging.getLogger(__name__)

PUBLICATION_SLUGS: dict[str, str] = {
    "gaias web": "gaias-web",
    "native state": "native-state",
    "knowing otherwise": "knowing-otherwise",
    "the arriving future": "the-arriving-future",
    "reasoning under uncertainty": "reasoning-under-uncertainty",
}


def _slugify_topic(topic: str, max_length: int = 80) -> str:
    """Convert a topic title to a kebab-case filename slug.

    Examples:
        "Forest Decline Dynamics: Drought, Fire, and Ecosystem Collapse"
        → "forest-decline-dynamics-drought-fire-and-ecosystem-collapse"
    """
    slug = topic.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)  # strip non-alphanumeric
    slug = re.sub(r"[\s]+", "-", slug.strip())  # spaces → hyphens
    slug = re.sub(r"-{2,}", "-", slug)  # collapse multiple hyphens
    return slug[:max_length].rstrip("-")


def _extract_abstract(content: str) -> str:
    """Extract the abstract from a lit review's content.

    The abstract is an italic block (``*...*``) appearing after the first H1.
    Falls back to an empty string if not found.
    """
    # Match a multi-line italic block: *text spanning lines*
    match = re.search(r"^\*(.+?)\*$", content, re.MULTILINE | re.DOTALL)
    if match:
        # Collapse internal whitespace to a single space
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _build_frontmatter(
    topic: str,
    description: str,
    date: str,
    publication_slug: str,
    quality: str,
) -> str:
    """Build YAML frontmatter for a Quartz lit review page."""
    # Escape quotes in title/description for YAML
    safe_title = topic.replace('"', '\\"')
    safe_desc = description.replace('"', '\\"')

    # Use just the date portion of an ISO timestamp
    date_str = date[:10] if len(date) >= 10 else date

    tags_block = "\n".join(
        [
            "tags:",
            "  - literature-review",
            f"  - {publication_slug}",
        ]
    )

    return "\n".join(
        [
            "---",
            f'title: "{safe_title}"',
            f'description: "{safe_desc}"',
            f"date: {date_str}",
            tags_block,
            f"quality: {quality}",
            "draft: false",
            "---",
        ]
    )


async def export_lit_review_to_quartz(
    content: str,
    topic: str,
    category: str,
    generated_at: str,
    quality: str,
) -> Path | None:
    """Export a literature review to the Quartz content directory.

    Args:
        content: The enhanced lit review markdown (final_report).
        topic: Review topic title.
        category: Publication category (e.g. "gaias web").
        generated_at: ISO timestamp for the date field.
        quality: Quality tier for metadata.

    Returns:
        Path to the written file, or None on failure.
    """
    pub_slug = PUBLICATION_SLUGS.get(category.lower())
    if not pub_slug:
        logger.warning("No publication slug for category %r, skipping Quartz export", category)
        return None

    topic_slug = _slugify_topic(topic)
    abstract = _extract_abstract(content)

    frontmatter = _build_frontmatter(
        topic=topic,
        description=abstract,
        date=generated_at,
        publication_slug=pub_slug,
        quality=quality,
    )

    pub_dir = QUARTZ_CONTENT_DIR / pub_slug
    pub_dir.mkdir(parents=True, exist_ok=True)

    out_path = pub_dir / f"{topic_slug}.md"
    out_path.write_text(f"{frontmatter}\n\n{content}\n")

    logger.info("Exported lit review to %s", out_path)
    return out_path
