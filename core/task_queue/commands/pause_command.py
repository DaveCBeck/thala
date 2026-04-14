"""CLI commands for pausing and resuming the parallel runner."""

from __future__ import annotations

import argparse

from core.task_queue.pause import (
    clear_paused,
    is_paused,
    read_paused_marker,
    set_paused,
)


def cmd_pause(args: argparse.Namespace) -> None:
    """Create the pause flag. Running workflows block at the next checkpoint."""
    reason = getattr(args, "reason", None)
    if is_paused():
        marker = read_paused_marker() or ""
        print(f"Already paused ({marker})")
        return
    set_paused(reason=reason)
    marker = read_paused_marker() or ""
    print(f"Paused. Workflows will block at the next checkpoint.\n  marker: {marker}")


def cmd_resume(args: argparse.Namespace) -> None:
    """Remove the pause flag. Blocked workflows resume within ~30 s."""
    del args
    cleared = clear_paused()
    if cleared:
        print("Resumed. Blocked workflows will pick up within ~30 s.")
    else:
        print("Not paused (no flag file present).")
