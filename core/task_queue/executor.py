"""Task execution utilities."""
import asyncio
from typing import TYPE_CHECKING

from .workflows import DEFAULT_WORKFLOW_TYPE

if TYPE_CHECKING:
    from .budget_tracker import BudgetTracker
    from .checkpoint import CheckpointManager
    from .queue_manager import TaskQueueManager


def _display_result(result: dict, task: dict) -> None:
    """Display workflow result to user."""
    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
    task_id = task["id"][:8]
    status = result.get("status", "unknown")

    if status == "success":
        print("\n✓ Workflow completed successfully")
        print(f"  Task: {task_id} ({task_type})")
    elif status == "partial":
        print("\n⚠ Workflow completed with errors")
        print(f"  Task: {task_id} ({task_type})")
        if result.get("errors"):
            print(f"  Errors: {result['errors']}")
    else:
        print("\n✗ Workflow failed")
        print(f"  Task: {task_id} ({task_type})")
        if result.get("errors"):
            print(f"  Errors: {result['errors']}")

    output_paths = result.get("output_paths", {})
    if output_paths:
        print("\n  Outputs:")
        for name, path in output_paths.items():
            print(f"    {name}: {path}")


def _run_workflow_sync(
    task: dict,
    manager: "TaskQueueManager",
    checkpoint_mgr: "CheckpointManager",
    budget_tracker: "BudgetTracker",
    resume_from: dict | None = None,
) -> int:
    """Run workflow synchronously and display results.

    Returns exit code (0=success, 1=failure, 130=interrupted).
    """
    from .runner import run_task_workflow

    async def _run():
        return await run_task_workflow(
            task=task,
            queue_manager=manager,
            checkpoint_mgr=checkpoint_mgr,
            budget_tracker=budget_tracker,
            resume_from=resume_from,
            shutdown_coordinator=None,
        )

    try:
        result = asyncio.run(_run())
        _display_result(result, task)
        return 0 if result.get("status") in ("success", "partial") else 1
    except KeyboardInterrupt:
        print("\n\nInterrupted. Checkpoint preserved for resumption.")
        return 130
    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        return 1
