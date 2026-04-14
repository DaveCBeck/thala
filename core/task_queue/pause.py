"""File-flag pause/resume for the parallel runner.

A simple mechanism to hold the runner at natural checkpoint boundaries
without killing the process. Creating ``.thala/queue/paused`` makes
``wait_if_paused()`` block; removing it lets blocked callers proceed.

Hook points:
* ``IncrementalStateManager.save_progress`` — after every supervision
  iteration save, so mid-iteration LLM work completes before blocking.
* ``run_task_workflow`` — before a task starts, so newly-selected tasks
  don't launch while paused.

The flag is checked with a poll loop rather than inotify/signals so that
the behaviour is identical across platforms and survives clock jumps.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from core.task_queue.paths import PAUSE_FLAG_FILE

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 30.0


def is_paused() -> bool:
    """True if the pause flag is currently set."""
    return PAUSE_FLAG_FILE.exists()


def set_paused(reason: str | None = None) -> None:
    """Create the pause flag. Idempotent.

    Writes an ISO timestamp (plus optional reason) so operators can see
    when the pause was requested.
    """
    PAUSE_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = datetime.now(timezone.utc).isoformat()
    if reason:
        payload += f" {reason}"
    PAUSE_FLAG_FILE.write_text(payload + "\n")


def clear_paused() -> bool:
    """Remove the pause flag. Returns True if the flag existed."""
    if PAUSE_FLAG_FILE.exists():
        PAUSE_FLAG_FILE.unlink()
        return True
    return False


def read_paused_marker() -> str | None:
    """Return the flag file's contents, or None if not paused."""
    if not PAUSE_FLAG_FILE.exists():
        return None
    try:
        return PAUSE_FLAG_FILE.read_text().strip()
    except OSError:
        return ""


async def wait_if_paused(label: str = "task-queue") -> None:
    """Block while the pause flag is set; return promptly if not paused.

    Polls every ``_POLL_INTERVAL_SECONDS``. Safe to call in hot paths —
    the only cost when unpaused is one ``Path.exists()`` stat.
    """
    if not is_paused():
        return

    marker = read_paused_marker() or ""
    logger.info(
        f"[{label}] paused at {datetime.now(timezone.utc).isoformat()} "
        f"(flag: {marker!r}); polling every {_POLL_INTERVAL_SECONDS:.0f}s"
    )

    while is_paused():
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    logger.info(f"[{label}] resumed")
