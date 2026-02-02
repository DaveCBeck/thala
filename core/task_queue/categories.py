"""Category management for task queue.

Loads categories from publications.json (source of truth).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback categories if publications.json doesn't exist
_FALLBACK_CATEGORIES = [
    "philosophy",
    "science",
    "technology",
    "society",
    "culture",
]


def load_categories_from_publications(publications_file: Path) -> list[str]:
    """Load category names from publications.json.

    publications.json is the source of truth for categories.
    Categories are derived from the top-level keys in the file.

    Args:
        publications_file: Path to publications.json

    Returns:
        List of category names
    """
    if publications_file.exists():
        try:
            with open(publications_file) as f:
                pubs = json.load(f)
            return list(pubs.keys())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load publications.json: {e}")

    return _FALLBACK_CATEGORIES


def get_default_categories(publications_file: Path) -> list[str]:
    """Get default categories from publications.json."""
    return load_categories_from_publications(publications_file)
