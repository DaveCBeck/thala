"""Run command implementation."""
import asyncio
import sys

from ..budget_tracker import BudgetTracker
from ..checkpoint import CheckpointManager
from ..executor import _run_workflow_sync
from ..pricing import format_cost
from ..queue_manager import TaskQueueManager
from ..workflows import DEFAULT_WORKFLOW_TYPE


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
    incomplete = asyncio.run(checkpoint_mgr.get_incomplete_work())
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

        print("\nResuming workflow...")
        sys.exit(_run_workflow_sync(
            task=task,
            manager=manager,
            checkpoint_mgr=checkpoint_mgr,
            budget_tracker=budget_tracker,
            resume_from=checkpoint,
        ))

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

    print("\nStarting workflow...")
    sys.exit(_run_workflow_sync(
        task=task,
        manager=manager,
        checkpoint_mgr=checkpoint_mgr,
        budget_tracker=budget_tracker,
        resume_from=None,
    ))
