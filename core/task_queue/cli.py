#!/usr/bin/env python3
"""
CLI for task queue management.

Usage:
    # Add tasks (default: lit_review_full)
    python -m core.task_queue.cli add "topic text" -c science
    python -m core.task_queue.cli add "topic text" -c science --type lit_review_full

    # Add web research task
    python -m core.task_queue.cli add "research query" -c technology --type web_research

    # Other commands
    python -m core.task_queue.cli list
    python -m core.task_queue.cli status
    python -m core.task_queue.cli run [-y]
    python -m core.task_queue.cli start      # Start daemon
    python -m core.task_queue.cli stop       # Stop daemon
    python -m core.task_queue.cli config --mode stagger_hours --stagger-hours 24
    python -m core.task_queue.cli reorder --export
"""
# ruff: noqa: E402  # Module imports after sys.path modification

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from core.config import configure_logging  # noqa: E402

from .commands import cmd_add, cmd_config, cmd_list, cmd_reorder, cmd_run, cmd_status  # noqa: E402
from .daemon import cmd_daemon, cmd_start, cmd_stop  # noqa: E402
from .workflows import DEFAULT_WORKFLOW_TYPE, get_available_types  # noqa: E402


def main():
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Task queue management for literature review workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add command
    add_parser = subparsers.add_parser("add", help="Add task to queue")
    add_parser.add_argument("topic", help="Topic text (or query for web_research)")
    add_parser.add_argument(
        "--type", "-t",
        choices=get_available_types(),
        default=DEFAULT_WORKFLOW_TYPE,
        help=f"Workflow type (default: {DEFAULT_WORKFLOW_TYPE})"
    )
    add_parser.add_argument(
        "--category", "-c", required=True,
        help="Task category (philosophy, science, technology, society, culture, or custom)"
    )
    add_parser.add_argument(
        "--priority", "-p", default="normal",
        help="Priority (low, normal, high, urgent)"
    )
    add_parser.add_argument(
        "--questions", "-q",
        help="Research questions for lit_review_full (pipe-separated, e.g., 'Q1|Q2|Q3')"
    )
    add_parser.add_argument(
        "--quality", default="standard",
        help="Quality tier (test, quick, standard, comprehensive, high_quality)"
    )
    add_parser.add_argument("--language", "-l", default="en", help="Language code")
    add_parser.add_argument("--from-year", type=int, help="Start year for papers (lit_review_full only)")
    add_parser.add_argument("--to-year", type=int, help="End year for papers (lit_review_full only)")
    add_parser.add_argument("--notes", help="Notes for this task")
    add_parser.add_argument("--tags", help="Tags (comma-separated)")
    add_parser.set_defaults(func=cmd_add)

    # list command
    list_parser = subparsers.add_parser("list", help="List queue")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.set_defaults(func=cmd_list)

    # run command
    run_parser = subparsers.add_parser("run", help="Run next eligible task")
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    run_parser.add_argument(
        "--skip-resume", action="store_true",
        help="Skip incomplete work and start fresh"
    )
    run_parser.set_defaults(func=cmd_run)

    # status command
    status_parser = subparsers.add_parser("status", help="Show status")
    status_parser.set_defaults(func=cmd_status)

    # reorder command
    reorder_parser = subparsers.add_parser("reorder", help="Reorder queue")
    reorder_parser.add_argument(
        "--export", "-e", action="store_true",
        help="Export current order as JSON"
    )
    reorder_parser.add_argument(
        "--input", "-i",
        help="Import new order from JSON file"
    )
    reorder_parser.set_defaults(func=cmd_reorder)

    # config command
    config_parser = subparsers.add_parser("config", help="Configure concurrency")
    config_parser.add_argument(
        "--mode", "-m",
        choices=["max_concurrent", "stagger_hours"],
        help="Concurrency mode"
    )
    config_parser.add_argument(
        "--max-concurrent", type=int,
        help="Max concurrent tasks (for max_concurrent mode)"
    )
    config_parser.add_argument(
        "--stagger-hours", type=float,
        help="Hours between task starts (for stagger_hours mode)"
    )
    config_parser.set_defaults(func=cmd_config)

    # start command
    start_parser = subparsers.add_parser("start", help="Start queue daemon")
    start_parser.set_defaults(func=cmd_start)

    # stop command
    stop_parser = subparsers.add_parser("stop", help="Stop queue daemon")
    stop_parser.set_defaults(func=cmd_stop)

    # daemon command (internal)
    daemon_parser = subparsers.add_parser("daemon", help="Run as daemon (internal)")
    daemon_parser.add_argument(
        "--max-tasks", type=int,
        help="Max tasks to process before stopping"
    )
    daemon_parser.add_argument(
        "--check-interval", type=float, default=300.0,
        help="Seconds between queue checks (default: 300)"
    )
    daemon_parser.set_defaults(func=cmd_daemon)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
