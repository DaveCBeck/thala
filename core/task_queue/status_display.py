"""
Queue and budget status display.

Provides:
- Print current queue statistics
- Print budget status
- Print active work
- Print next eligible task
"""

import asyncio
import logging

from .budget_tracker import BudgetTracker
from .checkpoint import CheckpointManager
from .pricing import format_cost
from .queue_manager import TaskQueueManager
from .workflows import DEFAULT_WORKFLOW_TYPE

logger = logging.getLogger(__name__)


async def print_status_async():
    """Print current queue and budget status (async implementation)."""
    queue_manager = TaskQueueManager()
    checkpoint_mgr = CheckpointManager()
    budget_tracker = BudgetTracker()

    # Budget status
    print("=== BUDGET ===")
    status = budget_tracker.get_budget_status()
    print(f"  Month-to-date: {format_cost(status['current_cost'])}")
    print(f"  Budget: {format_cost(status['monthly_budget'])}")
    print(f"  Used: {status['percent_used']:.1f}%")
    print(f"  Action: {status['action']}")

    # Queue status
    print("\n=== QUEUE ===")
    stats = queue_manager.get_queue_stats()
    for status_name, count in stats["by_status"].items():
        print(f"  {status_name}: {count}")

    # Active work
    active = await checkpoint_mgr.get_active_work()
    if active:
        print(f"\n=== ACTIVE ({len(active)}) ===")
        for cp in active:
            task_id = cp.get("task_id") or cp.get("topic_id", "unknown")
            task_type = cp.get("task_type", DEFAULT_WORKFLOW_TYPE)
            print(f"  {task_id[:8]} ({task_type}): {cp['phase']}")

    # Next eligible
    next_task = queue_manager.get_next_eligible_task()
    if next_task:
        print("\n=== NEXT ===")
        task_type = next_task.get("task_type", DEFAULT_WORKFLOW_TYPE)
        task_identifier = next_task.get("topic") or next_task.get("query", "unknown")
        print(f"  [{task_type}] {task_identifier[:50]}...")


def print_status():
    """Print current queue and budget status.

    Sync wrapper that runs the async implementation.
    """
    asyncio.run(print_status_async())
