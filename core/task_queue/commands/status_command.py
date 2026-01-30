"""Status command implementation."""
import asyncio

from ..budget_tracker import BudgetTracker
from ..checkpoint import CheckpointManager
from ..pricing import format_cost
from ..queue_manager import TaskPriority, TaskQueueManager, TaskStatus
from ..workflows import DEFAULT_WORKFLOW_TYPE


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
    incomplete = asyncio.run(checkpoint_mgr.get_incomplete_work())
    active = asyncio.run(checkpoint_mgr.get_active_work())
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
