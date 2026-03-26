"""Editorial stance utilities for evening_reads workflow.

Loads publication-specific editorial stances from .thala/editorial_stances/
to inject as intellectual priors into writing prompts.

A shared editorial identity file (_identity.md) is prepended to every
publication-specific stance, providing cross-cutting values.
"""

IDENTITY_FILENAME = "_identity.md"


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

    return _combine(_load_identity(), stance_file.read_text())


def load_editorial_stance_by_slug(publication_slug: str) -> str | None:
    """Load editorial stance file by publication slug directly.

    For use in testing where the publication slug is provided directly
    rather than derived from a task category.

    The shared editorial identity (_identity.md) is prepended when present.

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

    return _combine(_load_identity(), stance_file.read_text())
