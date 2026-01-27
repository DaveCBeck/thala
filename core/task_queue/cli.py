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
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from core.task_queue.budget_tracker import BudgetTracker  # noqa: E402
from core.task_queue.checkpoint_manager import CheckpointManager  # noqa: E402
from core.task_queue.pricing import format_cost  # noqa: E402
from core.task_queue.queue_manager import (  # noqa: E402
    TaskCategory,
    TaskPriority,
    TaskQueueManager,
    TaskStatus,
)
from core.task_queue.workflows import get_available_types, DEFAULT_WORKFLOW_TYPE  # noqa: E402


def cmd_add(args):
    """Add task to queue."""
    manager = TaskQueueManager()

    # Determine task type
    task_type = getattr(args, "type", None) or DEFAULT_WORKFLOW_TYPE

    # Validate task type
    available_types = get_available_types()
    if task_type not in available_types:
        print(f"Invalid task type: {task_type}")
        print(f"Valid: {', '.join(available_types)}")
        sys.exit(1)

    # Parse category
    try:
        # Try enum first, then as string
        categories = manager.get_categories()
        if args.category.upper() in [c.upper() for c in categories]:
            category = args.category.lower()
        elif args.category.upper() in [c.name for c in TaskCategory]:
            category = TaskCategory[args.category.upper()].value
        else:
            category = args.category.lower()  # Accept any string
    except Exception:
        category = args.category.lower()

    # Parse priority
    try:
        priority = TaskPriority[args.priority.upper()]
    except KeyError:
        print(f"Invalid priority: {args.priority}")
        print(f"Valid: {', '.join(p.name.lower() for p in TaskPriority)}")
        sys.exit(1)

    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # Build task based on type
    if task_type == "lit_review_full":
        # Parse research questions (lit_review only)
        questions = None
        if args.questions:
            questions = [q.strip() for q in args.questions.split("|") if q.strip()]

        # Parse date range (lit_review only)
        date_range = None
        if args.from_year or args.to_year:
            date_range = (args.from_year or 2000, args.to_year or 2026)

        task_id = manager.add_task(
            task_type=task_type,
            topic=args.topic,
            category=category,
            priority=priority,
            research_questions=questions,
            quality=args.quality,
            language=args.language,
            date_range=date_range,
            notes=args.notes,
            tags=tags,
        )
        identifier = args.topic

    elif task_type == "web_research":
        task_id = manager.add_task(
            task_type=task_type,
            query=args.topic,  # Reuse positional arg as query
            category=category,
            priority=priority,
            quality=args.quality,
            language=args.language if args.language != "en" else None,
            notes=args.notes,
            tags=tags,
        )
        identifier = args.topic

    else:
        # Generic fallback
        task_id = manager.add_task(
            task_type=task_type,
            topic=args.topic,
            category=category,
            priority=priority,
            quality=args.quality,
            notes=args.notes,
            tags=tags,
        )
        identifier = args.topic

    print(f"Added task: {task_id}")
    print(f"  Type: {task_type}")
    print(f"  {'Query' if task_type == 'web_research' else 'Topic'}: {identifier[:60]}{'...' if len(identifier) > 60 else ''}")
    print(f"  Category: {category}")
    print(f"  Priority: {priority.name}")
    print(f"  Quality: {args.quality}")


def cmd_list(args):
    """List tasks in queue."""
    manager = TaskQueueManager()

    status_filter = None
    if args.status:
        try:
            status_filter = TaskStatus[args.status.upper()]
        except KeyError:
            print(f"Invalid status: {args.status}")
            print(f"Valid: {', '.join(s.name.lower() for s in TaskStatus)}")
            sys.exit(1)

    category_filter = None
    if args.category:
        category_filter = args.category.lower()

    tasks = manager.list_tasks(status=status_filter, category=category_filter)

    if not tasks:
        print("Queue is empty")
        return

    # Group by status
    by_status: dict[str, list] = {}
    for task in tasks:
        status = task["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(task)

    # Print in status order
    status_order = ["in_progress", "pending", "paused", "completed", "failed"]
    for status in status_order:
        if status not in by_status:
            continue

        group = by_status[status]
        print(f"\n=== {status.upper()} ({len(group)}) ===")

        for task in group:
            try:
                priority = TaskPriority(task["priority"]).name
            except ValueError:
                priority = str(task["priority"])

            task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
            task_identifier = task.get("topic") or task.get("query", "unknown")
            type_badge = f"[{task_type}]" if task_type != DEFAULT_WORKFLOW_TYPE else ""

            print(f"  [{task['id'][:8]}] {type_badge} {task_identifier[:50]}...")
            print(f"           Category: {task['category']}, Priority: {priority}")

            if task.get("started_at"):
                print(f"           Started: {task['started_at'][:19]}")
            if task.get("current_phase"):
                print(f"           Phase: {task['current_phase']}")
            if task.get("error_message"):
                print(f"           Error: {task['error_message'][:50]}...")


def cmd_run(args):
    """Run next eligible task or resume in-progress work."""
    manager = TaskQueueManager()
    checkpoint_mgr = CheckpointManager()
    budget_tracker = BudgetTracker()

    # Check budget
    should_proceed, reason = budget_tracker.should_proceed()
    if not should_proceed:
        print(f"Cannot proceed: {reason}")
        sys.exit(1)

    # Check for incomplete work to resume
    incomplete = checkpoint_mgr.get_incomplete_work()
    if incomplete and not args.skip_resume:
        checkpoint = incomplete[0]
        task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")  # backward compat
        task = manager.get_task(task_id)

        task_type = checkpoint.get("task_type", DEFAULT_WORKFLOW_TYPE)
        print(f"Found incomplete work: {task_id[:8]} ({task_type})")
        if task:
            task_identifier = task.get("topic") or task.get("query", "unknown")
            print(f"  {'Query' if task_type == 'web_research' else 'Topic'}: {task_identifier[:50]}...")
        print(f"  Phase: {checkpoint['phase']}")

        # Show counters if available
        counters = checkpoint.get("counters", {})
        if counters:
            counter_str = ", ".join(f"{k}: {v}" for k, v in counters.items())
            print(f"  Progress: {counter_str}")

        print(f"  Last checkpoint: {checkpoint['last_checkpoint_at'][:19]}")

        resume_phase = checkpoint_mgr.can_resume_from_phase(checkpoint)
        print(f"  Will resume from: {resume_phase}")

        if not args.yes:
            response = input("\nResume this work? [y/N] ")
            if response.lower() != "y":
                sys.exit(0)

        # TODO: Invoke workflow with resume state
        print("\nResume functionality not yet implemented.")
        print("Use the runner module to execute workflows.")
        return

    # Get next eligible task
    task = manager.get_next_eligible_task()

    if not task:
        # Check why no task is eligible
        stats = manager.get_queue_stats()
        pending = stats["by_status"].get("pending", 0)
        in_progress = stats["by_status"].get("in_progress", 0)

        print("No eligible tasks to run")
        if pending == 0:
            print("  (No pending tasks in queue)")
        elif in_progress > 0:
            config = stats["concurrency"]
            if config["mode"] == "stagger_hours":
                print(f"  (Waiting for stagger period: {config['stagger_hours']} hours)")
            else:
                print(f"  (Max concurrent reached: {in_progress}/{config['max_concurrent']})")
        sys.exit(0)

    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
    task_identifier = task.get("topic") or task.get("query", "unknown")

    print(f"Selected task: {task['id'][:8]} ({task_type})")
    print(f"  {'Query' if task_type == 'web_research' else 'Topic'}: {task_identifier[:60]}...")
    print(f"  Category: {task['category']}")
    print(f"  Quality: {task['quality']}")

    # Show budget status
    status = budget_tracker.get_budget_status()
    print(f"\n  Budget: {format_cost(status['current_cost'])} / {format_cost(status['monthly_budget'])} "
          f"({status['percent_used']:.1f}%)")

    if not args.yes:
        response = input("\nStart this task? [y/N] ")
        if response.lower() != "y":
            sys.exit(0)

    # TODO: Invoke workflow
    print("\nWorkflow execution not yet implemented.")
    print("Use the runner module to execute workflows.")


def cmd_status(args):
    """Show current status: work, budget, next scheduled."""
    manager = TaskQueueManager()
    checkpoint_mgr = CheckpointManager()
    budget_tracker = BudgetTracker()

    # Budget status
    print("=== BUDGET STATUS ===")
    try:
        status = budget_tracker.get_budget_status(show_progress=True)
        print(f"  Month-to-date: {format_cost(status['current_cost'])}")
        print(f"  Monthly budget: {format_cost(status['monthly_budget'])}")
        print(f"  Used: {status['percent_used']:.1f}%")
        print(f"  Remaining: {format_cost(status['remaining'])}")
        print(f"  Days left: {status['days_remaining']}")
        print(f"  Daily budget: {format_cost(status['daily_budget_remaining'])}")
        print(f"  Action: {status['action']}")
    except Exception as e:
        print(f"  Error getting budget: {e}")
        import traceback
        traceback.print_exc()
        status = None

    # Current work
    print("\n=== CURRENT WORK ===")
    incomplete = checkpoint_mgr.get_incomplete_work()
    active = checkpoint_mgr.get_active_work()
    in_progress = manager.list_tasks(status=TaskStatus.IN_PROGRESS)

    if not in_progress and not incomplete and not active:
        print("  No tasks currently running")
    else:
        for task in in_progress:
            checkpoint = checkpoint_mgr.get_checkpoint(task["id"])
            task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
            task_identifier = task.get("topic") or task.get("query", "unknown")
            print(f"  [{task['id'][:8]}] ({task_type}) {task_identifier[:40]}...")
            if checkpoint:
                print(f"           Phase: {checkpoint['phase']}")
                counters = checkpoint.get("counters", {})
                if counters:
                    counter_str = ", ".join(f"{k}: {v}" for k, v in counters.items())
                    print(f"           Progress: {counter_str}")

        if incomplete:
            print("\n  Incomplete (resumable):")
            for cp in incomplete:
                task_id = cp.get("task_id") or cp.get("topic_id", "unknown")
                task_type = cp.get("task_type", DEFAULT_WORKFLOW_TYPE)
                print(f"    [{task_id[:8]}] ({task_type}) Phase: {cp['phase']}")

    # Queue summary
    print("\n=== QUEUE SUMMARY ===")
    stats = manager.get_queue_stats()
    for status_name, count in stats["by_status"].items():
        print(f"  {status_name}: {count}")
    print(f"  Total: {stats['total']}")

    # Concurrency config
    config = stats["concurrency"]
    print(f"\n  Concurrency mode: {config['mode']}")
    if config["mode"] == "stagger_hours":
        print(f"  Stagger: {config['stagger_hours']} hours")
        if status:
            adaptive = budget_tracker.get_adaptive_stagger_hours(config["stagger_hours"])
            if adaptive != config["stagger_hours"]:
                print(f"  Adaptive stagger: {adaptive:.1f} hours")
    else:
        print(f"  Max concurrent: {config['max_concurrent']}")

    # Next eligible
    print("\n=== NEXT ELIGIBLE ===")
    next_task = manager.get_next_eligible_task()
    if next_task:
        task_type = next_task.get("task_type", DEFAULT_WORKFLOW_TYPE)
        task_identifier = next_task.get("topic") or next_task.get("query", "unknown")
        print(f"  [{next_task['id'][:8]}] ({task_type}) {task_identifier[:40]}...")
        print(f"           Category: {next_task['category']}")
        print(f"           Priority: {TaskPriority(next_task['priority']).name}")
    else:
        print("  No tasks eligible (check concurrency constraints)")


def cmd_reorder(args):
    """Reorder/export queue for LLM editing."""
    manager = TaskQueueManager()

    pending = manager.list_tasks(status=TaskStatus.PENDING)

    if not pending:
        print("No pending tasks to reorder")
        return

    # Export current order in LLM-friendly format
    if args.export:
        data = {
            "instructions": "Reorder tasks by editing the 'order' list. Highest priority first. "
                          "You can also edit priority values (1=low, 2=normal, 3=high, 4=urgent).",
            "tasks": [
                {
                    "id": t["id"],
                    "topic": t["topic"],
                    "category": t["category"],
                    "priority": TaskPriority(t["priority"]).name,
                    "notes": t.get("notes"),
                }
                for t in pending
            ],
            "order": [t["id"] for t in pending],
        }
        print(json.dumps(data, indent=2))
        return

    # Import new order
    if args.input:
        with open(args.input, "r") as f:
            data = json.load(f)

        new_order = data.get("order", [])
        manager.reorder(new_order)
        print(f"Reordered {len(new_order)} tasks")
        return

    # No action specified - show current order
    print("Current task order (pending only):")
    for i, task in enumerate(pending, 1):
        priority = TaskPriority(task["priority"]).name
        print(f"  {i}. [{task['id'][:8]}] {task['topic'][:40]}... ({priority})")

    print("\nUse --export to export for editing, --input FILE to import new order")


def cmd_config(args):
    """Configure concurrency settings."""
    manager = TaskQueueManager()

    if args.mode:
        manager.set_concurrency(
            mode=args.mode,
            max_concurrent=args.max_concurrent or 1,
            stagger_hours=args.stagger_hours or 36.0,
        )
        print(f"Set concurrency mode: {args.mode}")
        if args.mode == "stagger_hours":
            print(f"  Stagger: {args.stagger_hours or 36.0} hours")
        else:
            print(f"  Max concurrent: {args.max_concurrent or 1}")
    else:
        # Show current config
        config = manager.get_concurrency_config()
        print("Current concurrency config:")
        print(f"  Mode: {config['mode']}")
        print(f"  Max concurrent: {config['max_concurrent']}")
        print(f"  Stagger hours: {config['stagger_hours']}")


# Daemon management (project root / topic_queue)
DAEMON_PID_FILE = PROJECT_ROOT / "topic_queue" / "daemon.pid"


def _get_daemon_pid() -> int | None:
    """Get PID of running daemon, or None if not running."""
    if not DAEMON_PID_FILE.exists():
        return None

    try:
        pid = int(DAEMON_PID_FILE.read_text().strip())
        # Check if process is alive
        import os
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        # PID file is stale
        DAEMON_PID_FILE.unlink(missing_ok=True)
        return None


def cmd_start(args):
    """Start the queue daemon."""
    import os
    import subprocess

    pid = _get_daemon_pid()
    if pid:
        print(f"Daemon already running (PID {pid})")
        return

    # Start daemon in background
    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.task_queue.cli", "daemon"],
        stdout=open(Path.home() / ".thala" / "topic_queue" / "daemon.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )

    print(f"Started daemon (PID {proc.pid})")
    print("  Log: ~/.thala/topic_queue/daemon.log")
    print("  Stop with: python -m core.task_queue.cli stop")


def cmd_stop(args):
    """Stop the queue daemon."""
    import os
    import signal

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
    """Run as daemon (internal use)."""
    import asyncio
    import signal

    # Write PID file
    DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    DAEMON_PID_FILE.write_text(str(os.getpid()))

    print(f"Daemon started (PID {os.getpid()})")

    # Handle shutdown signals
    def shutdown(signum, frame):
        print("Shutting down...")
        DAEMON_PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Import runner and start loop
    from core.task_queue.runner import run_queue_loop

    try:
        asyncio.run(run_queue_loop(
            max_tasks=args.max_tasks,
            check_interval=args.check_interval,
        ))
    finally:
        DAEMON_PID_FILE.unlink(missing_ok=True)


def main():
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
