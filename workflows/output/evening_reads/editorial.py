"""Editorial stance utilities for evening_reads workflow.

Loads publication-specific editorial stances from .thala/editorial_stances/
to inject as intellectual priors into writing prompts.
"""

def load_editorial_stance(category: str) -> str | None:
    """Load editorial stance file for a category.

    Maps category names to stance files via slugification:
    - "Reasoning Under Uncertainty" -> reasoning-under-uncertainty.md
    - "Knowing Otherwise" -> knowing-otherwise.md

    Args:
        category: Task category (e.g., "Reasoning Under Uncertainty")

    Returns:
        Editorial stance markdown content, or None if not found
    """
    from core.task_queue.paths import EDITORIAL_STANCES_DIR

    # Slugify category to match file name
    slug = category.lower().replace(" ", "-")
    stance_file = EDITORIAL_STANCES_DIR / f"{slug}.md"

    if stance_file.exists():
        return stance_file.read_text()
    return None


def load_editorial_stance_by_slug(publication_slug: str) -> str | None:
    """Load editorial stance file by publication slug directly.

    For use in testing where the publication slug is provided directly
    rather than derived from a task category.

    Args:
        publication_slug: Publication slug (e.g., "reasoning-under-uncertainty")

    Returns:
        Editorial stance markdown content, or None if not found
    """
    from core.task_queue.paths import EDITORIAL_STANCES_DIR

    stance_file = EDITORIAL_STANCES_DIR / f"{publication_slug}.md"

    if stance_file.exists():
        return stance_file.read_text()
    return None
