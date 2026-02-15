"""Shared helper for loading Substack publication config.

Reads publications.json and returns the entry matching the given category,
falling back to the first entry if no exact match is found.
"""

import json
import logging

from core.task_queue.paths import PUBLICATIONS_FILE

logger = logging.getLogger(__name__)


def load_publication_config(category: str) -> dict:
    """Load publication config for a category from publications.json.

    Args:
        category: Task category key (e.g. "technology", "science").

    Returns:
        Dict with publication_url, subdomain, etc.  Empty dict on
        missing file or parse error.
    """
    if not PUBLICATIONS_FILE.exists():
        logger.warning("Publications config not found: %s", PUBLICATIONS_FILE)
        return {}

    try:
        with open(PUBLICATIONS_FILE) as f:
            pubs = json.load(f)
        if category in pubs:
            return pubs[category]
        if pubs:
            return next(iter(pubs.values()))
    except Exception:
        logger.warning("Failed to load publication config from %s", PUBLICATIONS_FILE, exc_info=True)

    return {}
