"""Per-publication sequential ID counter.

Manages a simple JSON counter file at ~/.thala/state/pub_counters.json,
keyed by publication subdomain (e.g. "arrivingfuture" → 5).
"""

import json
import logging

from pathlib import Path

from core.task_queue.paths import PUB_COUNTERS_FILE
from core.task_queue.utils import write_json_atomic

logger = logging.getLogger(__name__)


def _load_counters(path: Path | None = None) -> dict[str, int]:
    p = path or PUB_COUNTERS_FILE
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def next_id(pub_slug: str, *, counters_path: Path | None = None) -> int:
    """Atomically claim the next sequential ID for a publication.

    Returns the new ID (1-based).
    """
    p = counters_path or PUB_COUNTERS_FILE
    counters = _load_counters(p)
    new_id = counters.get(pub_slug, 0) + 1
    counters[pub_slug] = new_id
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, counters, indent=2)
    logger.info("Assigned %s seq_id=%d", pub_slug, new_id)
    return new_id


def peek_id(pub_slug: str, *, counters_path: Path | None = None) -> int:
    """Return the next ID that would be assigned, without incrementing."""
    counters = _load_counters(counters_path)
    return counters.get(pub_slug, 0) + 1
