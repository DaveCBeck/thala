"""Run command implementation."""

import asyncio
import sys

from ..budget_tracker import BudgetTracker
from ..parallel import run_parallel_tasks
from ..pricing import format_cost


def cmd_run(args):
    """Run next eligible task (wrapper around parallel dispatcher with count=1)."""
    budget_tracker = BudgetTracker()

    # Check budget
    should_proceed, reason = budget_tracker.should_proceed()
    if not should_proceed:
        print(f"Cannot proceed: {reason}")
        sys.exit(1)

    # Show budget status
    status = budget_tracker.get_budget_status()
    print(
        f"Budget: {format_cost(status['current_cost'])} / {format_cost(status['monthly_budget'])} "
        f"({status['percent_used']:.1f}%)"
    )

    if not args.yes:
        response = input("\nRun next eligible task? [y/N] ")
        if response.lower() != "y":
            sys.exit(0)

    print("\nStarting workflow...")
    try:
        results = asyncio.run(run_parallel_tasks(count=1, stagger_minutes=0))
    except KeyboardInterrupt:
        print("\n\nInterrupted. Checkpoint preserved for resumption.")
        sys.exit(130)

    if not results:
        print("No eligible tasks to run")
        sys.exit(0)

    result = results[0]
    if isinstance(result, BaseException):
        print(f"\nWorkflow failed: {result}")
        sys.exit(1)

    result_status = result.get("status", "unknown")
    if result_status in ("success", "partial"):
        print(f"\nWorkflow completed: {result_status}")
    elif result_status == "skipped":
        print(f"\nTask skipped: {result.get('reason', 'unknown')}")
    else:
        print(f"\nWorkflow result: {result_status}")
        sys.exit(1)
