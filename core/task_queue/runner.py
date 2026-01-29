"""
Workflow runner with checkpoint and budget integration.

Provides:
- Run tasks from queue with checkpoint callbacks
- Registry-based dispatch to workflow implementations
- Resume incomplete work
- Budget-aware execution with adaptive delays
- Continuous queue processing loop

Uses THALA_QUEUE_PROJECT for LangSmith tracing to isolate queue costs
from manual testing/development costs.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from .budget_tracker import BudgetTracker
from .checkpoint_manager import CheckpointManager
from .pricing import format_cost
from .queue_manager import TaskQueueManager
from .schemas import Task, WorkflowCheckpoint
from .workflows import get_workflow, DEFAULT_WORKFLOW_TYPE

logger = logging.getLogger(__name__)

# Queue uses a dedicated LangSmith project for budget isolation
QUEUE_PROJECT = os.getenv("THALA_QUEUE_PROJECT", "thala-queue")


def _find_bypass_task(queue_manager: TaskQueueManager) -> Optional[Task]:
    """Find a pending task that bypasses concurrency limits.

    Bypass tasks (like publish_series) can run regardless of stagger_hours
    or max_concurrent settings.

    Args:
        queue_manager: Queue manager instance

    Returns:
        First eligible bypass task, or None
    """
    from .schemas import TaskStatus

    pending_tasks = queue_manager.list_tasks(status=TaskStatus.PENDING)

    for task in pending_tasks:
        task_type = task.get("task_type", "lit_review_full")
        workflow = get_workflow(task_type)
        if getattr(workflow, "bypass_concurrency", False):
            return task

    return None


async def run_task_workflow(
    task: Task,
    queue_manager: TaskQueueManager,
    checkpoint_mgr: CheckpointManager,
    budget_tracker: BudgetTracker,
    resume_from: Optional[WorkflowCheckpoint] = None,
) -> dict:
    """Run a workflow for any task type via registry dispatch.

    Dispatches to the appropriate workflow based on task_type field.
    Handles checkpoint updates, budget checks, and status management.

    Args:
        task: Task to process (TopicTask, WebResearchTask, etc.)
        queue_manager: Queue manager instance
        checkpoint_mgr: Checkpoint manager instance
        budget_tracker: Budget tracker instance
        resume_from: Optional checkpoint to resume from

    Returns:
        Workflow result dict with status, outputs, etc.
    """
    # Set LangSmith project for queue runs (isolates budget from manual testing)
    os.environ["LANGSMITH_PROJECT"] = QUEUE_PROJECT
    logger.info(f"Using LangSmith project: {QUEUE_PROJECT}")

    # Get task type and workflow
    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
    workflow = get_workflow(task_type)

    task_id = task["id"]
    task_identifier = workflow.get_task_identifier(task)

    # Generate langsmith_run_id (or use existing if resuming)
    if resume_from:
        langsmith_run_id = resume_from["langsmith_run_id"]
        logger.info(f"Resuming task {task_id[:8]} from phase {resume_from['phase']}")
    else:
        langsmith_run_id = str(uuid.uuid4())
        logger.info(f"Starting task {task_id[:8]} ({task_type}): {task_identifier}")

    # Mark as started
    queue_manager.mark_started(task_id, langsmith_run_id)
    checkpoint_mgr.start_work(task_id, task_type, langsmith_run_id)

    try:
        # Create checkpoint callback
        def checkpoint_callback(phase: str, phase_outputs: dict = None, **kwargs):
            """Update checkpoint during workflow execution."""
            checkpoint_mgr.update_checkpoint(
                task_id, phase, phase_outputs=phase_outputs, **kwargs
            )
            queue_manager.update_phase(task_id, phase)

            # Check budget between phases
            should_proceed, reason = budget_tracker.should_proceed()
            if not should_proceed:
                logger.warning(f"Budget limit reached during {phase}: {reason}")
                # Don't raise - let the current phase complete

        # Run the workflow
        result = await workflow.run(task, checkpoint_callback, resume_from)

        # Save outputs
        checkpoint_callback("saving")
        output_paths = workflow.save_outputs(task, result)
        result["output_paths"] = output_paths

        if output_paths:
            logger.info(f"Saved outputs: {list(output_paths.keys())}")

        # Mark complete or failed
        if result.get("status") == "success":
            queue_manager.mark_completed(task_id)
            checkpoint_mgr.complete_work(task_id)
            logger.info(f"Task {task_id[:8]} completed successfully")
        elif result.get("status") == "partial":
            # Partial success - mark complete but log warnings
            queue_manager.mark_completed(task_id)
            checkpoint_mgr.complete_work(task_id)
            logger.warning(f"Task {task_id[:8]} completed with errors: {result.get('errors')}")
        else:
            error = str(result.get("errors", "Unknown error"))
            queue_manager.mark_failed(task_id, error)
            checkpoint_mgr.fail_work(task_id)
            logger.error(f"Task {task_id[:8]} failed: {error}")

        return result

    except Exception as e:
        logger.error(f"Task {task_id[:8]} failed with exception: {e}")
        queue_manager.mark_failed(task_id, str(e))
        checkpoint_mgr.fail_work(task_id)
        raise


async def run_single_task(
    queue_dir: Optional[Path] = None,
    skip_resume: bool = False,
    dry_run: bool = False,
) -> Optional[dict]:
    """Run the next eligible task from the queue.

    Args:
        queue_dir: Override queue directory (for testing)
        skip_resume: Skip incomplete work
        dry_run: If True, only show what would run

    Returns:
        Workflow result dict, or None if nothing to run
    """
    queue_manager = TaskQueueManager(queue_dir=queue_dir)
    checkpoint_mgr = CheckpointManager(queue_dir=queue_dir)
    budget_tracker = BudgetTracker(queue_dir=queue_dir)

    # Check for incomplete work first
    if not skip_resume:
        incomplete = checkpoint_mgr.get_incomplete_work()
        if incomplete:
            checkpoint = incomplete[0]
            task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")  # backward compat
            task = queue_manager.get_task(task_id)

            if task:
                task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
                task_identifier = task.get("topic") or task.get("query", "unknown")
                logger.info(f"Resuming incomplete work: {task_id[:8]} ({task_type})")

                if dry_run:
                    logger.info(f"Would resume: {task_identifier[:50]}...")
                    return None

                return await run_task_workflow(
                    task,
                    queue_manager,
                    checkpoint_mgr,
                    budget_tracker,
                    resume_from=checkpoint,
                )

    # First check for bypass tasks (ignore concurrency limits)
    task = _find_bypass_task(queue_manager)
    if task:
        task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
        logger.info(f"Found bypass task: {task_type} (ignoring concurrency limits)")
    else:
        # Get next eligible task (respects concurrency)
        task = queue_manager.get_next_eligible_task()

    if not task:
        logger.info("No eligible tasks to run")
        return None

    task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
    task_identifier = task.get("topic") or task.get("query", "unknown")

    # Check if zero-cost workflow (skip budget check)
    workflow = get_workflow(task_type)
    is_zero_cost = getattr(workflow, "is_zero_cost", False)

    if not is_zero_cost:
        # Check budget for workflows that incur costs
        should_proceed, reason = budget_tracker.should_proceed()
        if not should_proceed:
            logger.warning(f"Cannot proceed: {reason}")
            return None
    else:
        logger.info(f"Skipping budget check for zero-cost workflow: {task_type}")

    if dry_run:
        logger.info(f"Would run ({task_type}): {task_identifier[:50]}...")
        return None

    # Run the task
    return await run_task_workflow(
        task,
        queue_manager,
        checkpoint_mgr,
        budget_tracker,
    )


async def run_queue_loop(
    queue_dir: Optional[Path] = None,
    max_tasks: Optional[int] = None,
    dry_run: bool = False,
    check_interval: float = 300.0,  # 5 minutes
) -> None:
    """Continuously process tasks from queue.

    Args:
        queue_dir: Override queue directory (for testing)
        max_tasks: Maximum tasks to process (None = unlimited)
        dry_run: If True, only show what would run
        check_interval: Seconds between queue checks when nothing to run
    """
    queue_manager = TaskQueueManager(queue_dir=queue_dir)
    checkpoint_mgr = CheckpointManager(queue_dir=queue_dir)
    budget_tracker = BudgetTracker(queue_dir=queue_dir)

    tasks_processed = 0

    while True:
        # Check for incomplete work first
        incomplete = checkpoint_mgr.get_incomplete_work()
        if incomplete:
            checkpoint = incomplete[0]
            task_id = checkpoint.get("task_id") or checkpoint.get("topic_id")  # backward compat
            task = queue_manager.get_task(task_id)

            if task:
                task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
                logger.info(f"Resuming incomplete work: {task_id[:8]} ({task_type})")

                if not dry_run:
                    await run_task_workflow(
                        task,
                        queue_manager,
                        checkpoint_mgr,
                        budget_tracker,
                        resume_from=checkpoint,
                    )
                    tasks_processed += 1

                    if max_tasks and tasks_processed >= max_tasks:
                        logger.info(f"Processed {tasks_processed} tasks, stopping")
                        break
                continue

        # First check for bypass tasks (ignore concurrency limits)
        task = _find_bypass_task(queue_manager)
        if task:
            task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
            logger.info(f"Found bypass task: {task_type} (ignoring concurrency limits)")
        else:
            # Get next eligible task (respects concurrency)
            task = queue_manager.get_next_eligible_task()

        if not task:
            logger.info(f"No eligible tasks. Checking again in {check_interval}s...")
            await asyncio.sleep(check_interval)
            continue

        task_type = task.get("task_type", DEFAULT_WORKFLOW_TYPE)
        task_identifier = task.get("topic") or task.get("query", "unknown")

        # Check if zero-cost workflow (skip budget check)
        workflow = get_workflow(task_type)
        is_zero_cost = getattr(workflow, "is_zero_cost", False)

        if not is_zero_cost:
            # Check budget for workflows that incur costs
            should_proceed, reason = budget_tracker.should_proceed()
            if not should_proceed:
                logger.warning(f"Pausing due to budget: {reason}")
                # Sleep for an hour and retry
                await asyncio.sleep(3600)
                continue
        else:
            logger.info(f"Skipping budget check for zero-cost workflow: {task_type}")

        if dry_run:
            logger.info(f"Would run ({task_type}): {task_identifier[:50]}...")
            break

        # Run the task
        try:
            await run_task_workflow(
                task,
                queue_manager,
                checkpoint_mgr,
                budget_tracker,
            )
            tasks_processed += 1
        except Exception as e:
            logger.error(f"Task failed: {e}")
            # Continue to next task

        if max_tasks and tasks_processed >= max_tasks:
            logger.info(f"Processed {tasks_processed} tasks, stopping")
            break

        # Get adaptive delay based on budget (skip for zero-cost workflows)
        if not is_zero_cost:
            config = queue_manager.get_concurrency_config()
            if config["mode"] == "stagger_hours":
                base_hours = config["stagger_hours"]
                adaptive_hours = budget_tracker.get_adaptive_stagger_hours(base_hours)
                sleep_seconds = adaptive_hours * 3600

                logger.info(
                    f"Sleeping {adaptive_hours:.1f} hours before next task "
                    f"(base: {base_hours}h, budget-adjusted)"
                )
                await asyncio.sleep(sleep_seconds)


def print_status():
    """Print current queue and budget status."""
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
    active = checkpoint_mgr.get_active_work()
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
