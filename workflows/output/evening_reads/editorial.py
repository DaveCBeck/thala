"""Editorial stance utilities for evening_reads workflow.

Loads publication-specific editorial stances from .thala/editorial_stances/
to inject as intellectual priors into writing prompts.

A shared editorial identity file (_identity.md) is prepended to every
publication-specific stance, providing cross-cutting values.
"""

import yaml

IDENTITY_FILENAME = "_identity.md"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from text.

    Returns (frontmatter_dict, remaining_text) where remaining_text has
    the frontmatter block removed. Returns ({}, text) if no frontmatter.
    """
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    raw_yaml = text[4:end]
    try:
        parsed = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError:
        return {}, text

    remaining = text[end + 5:]  # skip past closing ---\n
    return parsed, remaining


def _load_identity() -> str:
    """Load the shared editorial identity, or empty string if not found."""
    from core.task_queue.paths import EDITORIAL_STANCES_DIR

    identity_file = EDITORIAL_STANCES_DIR / IDENTITY_FILENAME
    if identity_file.exists():
        return identity_file.read_text()
    return ""


def _combine(identity: str, stance: str) -> str:
    """Combine identity and publication stance with a separator."""
    if identity and stance:
        return identity.rstrip() + "\n\n---\n\n" + stance.lstrip()
    return identity or stance


def load_editorial_stance(category: str) -> str | None:
    """Load editorial stance file for a category.

    Maps category names to stance files via slugification:
    - "Reasoning Under Uncertainty" -> reasoning-under-uncertainty.md
    - "Knowing Otherwise" -> knowing-otherwise.md

    The shared editorial identity (_identity.md) is prepended when present.
    Frontmatter is stripped from the returned text.

    Args:
        category: Task category (e.g., "Reasoning Under Uncertainty")

    Returns:
        Editorial stance markdown content, or None if not found
    """
    from core.task_queue.paths import EDITORIAL_STANCES_DIR

    slug = category.lower().replace(" ", "-")
    stance_file = EDITORIAL_STANCES_DIR / f"{slug}.md"

    if not stance_file.exists():
        # Still return identity alone if it exists
        identity = _load_identity()
        return identity or None

    _, body = _parse_frontmatter(stance_file.read_text())
    return _combine(_load_identity(), body)


def load_editorial_stance_by_slug(publication_slug: str) -> str | None:
    """Load editorial stance file by publication slug directly.

    For use in testing where the publication slug is provided directly
    rather than derived from a task category.

    The shared editorial identity (_identity.md) is prepended when present.
    Frontmatter is stripped from the returned text.

    Args:
        publication_slug: Publication slug (e.g., "reasoning-under-uncertainty")

    Returns:
        Editorial stance markdown content, or None if not found
    """
    from core.task_queue.paths import EDITORIAL_STANCES_DIR

    stance_file = EDITORIAL_STANCES_DIR / f"{publication_slug}.md"

    if not stance_file.exists():
        identity = _load_identity()
        return identity or None

    _, body = _parse_frontmatter(stance_file.read_text())
    return _combine(_load_identity(), body)


def load_editorial_emphasis(slug: str) -> dict:
    """Load frontmatter emphasis config for a publication slug.

    Args:
        slug: Publication slug (e.g., "reasoning-under-uncertainty")

    Returns:
        Frontmatter dict (e.g., {"recency": "high"}), or {} if not found
    """
    from core.task_queue.paths import EDITORIAL_STANCES_DIR

    stance_file = EDITORIAL_STANCES_DIR / f"{slug}.md"
    if not stance_file.exists():
        return {}

    frontmatter, _ = _parse_frontmatter(stance_file.read_text())
    return frontmatter
