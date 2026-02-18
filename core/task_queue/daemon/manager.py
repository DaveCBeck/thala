"""Daemon process management."""

import asyncio
import os
import signal
import subprocess
import sys

from ..paths import DAEMON_LOG_FILE, DAEMON_PID_FILE


def _get_daemon_pid() -> int | None:
    """Get PID of running daemon, or None if not running."""
    if not DAEMON_PID_FILE.exists():
        return None

    try:
        pid = int(DAEMON_PID_FILE.read_text().strip())
        # Check if process is alive
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        # PID file is stale
        DAEMON_PID_FILE.unlink(missing_ok=True)
        return None


def cmd_start(args):
    """Start the queue daemon."""
    pid = _get_daemon_pid()
    if pid:
        print(f"Daemon already running (PID {pid})")
        return

    # Start daemon in background
    env = os.environ.copy()
    # Ensure parent directory exists
    DAEMON_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.task_queue.cli", "daemon"],
        stdout=open(DAEMON_LOG_FILE, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    print(f"Started daemon (PID {proc.pid})")
    print(f"  Log: {DAEMON_LOG_FILE}")
    print("  Stop with: python -m core.task_queue.cli stop")


def cmd_stop(args):
    """Stop the queue daemon."""
    pid = _get_daemon_pid()
    if not pid:
        print("Daemon not running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped daemon (PID {pid})")
        DAEMON_PID_FILE.unlink(missing_ok=True)
    except OSError as e:
        print(f"Failed to stop daemon: {e}")


def cmd_daemon(args):
    """Run as daemon (internal use).

    Wraps the parallel dispatcher in a continuous loop.
    Signal handlers for graceful shutdown are installed by run_daemon_loop().
    """
    # Write PID file
    DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAEMON_PID_FILE.write_text(str(os.getpid()))

    print(f"Daemon started (PID {os.getpid()})")

    from ..parallel import run_daemon_loop

    try:
        asyncio.run(
            run_daemon_loop(
                count=getattr(args, "count", 5),
                check_interval=args.check_interval,
                max_batches=args.max_tasks,
            )
        )
    finally:
        DAEMON_PID_FILE.unlink(missing_ok=True)
        print("Daemon stopped")
