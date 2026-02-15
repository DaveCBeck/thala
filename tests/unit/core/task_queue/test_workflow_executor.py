"""Tests for workflow_executor.py status handling.

Covers DEFERRED and "waiting" result status handling.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.task_queue.schemas.enums import TaskStatus


def _make_task(task_id="test-task-id", task_type="lit_review_full"):
    return {
        "id": task_id,
        "task_type": task_type,
        "topic": "Test topic",
        "category": "science",
        "priority": 2,
        "status": "in_progress",
        "quality": "standard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "langsmith_run_id": None,
        "current_phase": None,
        "error_message": None,
        "notes": None,
        "tags": [],
        "research_questions": None,
        "language": "en",
        "date_range": None,
    }


def _mock_workflow(result: dict, task_type="lit_review_full"):
    """Create a mock workflow that returns the given result."""
    workflow = MagicMock()
    workflow.task_type = task_type
    workflow.get_task_identifier.return_value = "test"
    workflow.save_outputs.return_value = {}

    async def mock_run(task, checkpoint_callback, resume_from, **kwargs):
        return result

    workflow.run = mock_run
    return workflow


@pytest.fixture
def mock_deps():
    """Common mocking for workflow executor dependencies."""
    queue_manager = MagicMock()
    queue_manager.mark_started = MagicMock()
    queue_manager.mark_completed = MagicMock()
    queue_manager.mark_failed = MagicMock()
    queue_manager.update_phase = MagicMock()
    queue_manager.update_task = MagicMock(return_value=True)

    checkpoint_mgr = MagicMock()
    checkpoint_mgr.start_work = AsyncMock()
    checkpoint_mgr.complete_work = AsyncMock()
    checkpoint_mgr.fail_work = AsyncMock()
    checkpoint_mgr.update_checkpoint = AsyncMock()

    budget_tracker = MagicMock()
    budget_tracker.should_proceed.return_value = (True, "")

    return queue_manager, checkpoint_mgr, budget_tracker


@pytest.mark.asyncio
async def test_deferred_status_updates_task_and_clears_checkpoint(mock_deps):
    """When workflow returns 'deferred', task gets DEFERRED status and checkpoint is cleared."""
    queue_manager, checkpoint_mgr, budget_tracker = mock_deps
    task = _make_task()

    next_run = "2026-02-16T12:00:00+00:00"
    workflow = _mock_workflow({"status": "deferred", "next_run_after": next_run})

    with (
        patch("core.task_queue.workflow_executor.get_workflow", return_value=workflow),
        patch("core.task_queue.workflow_executor.start_run"),
        patch("core.task_queue.workflow_executor.end_run"),
        patch("core.task_queue.workflow_executor.set_task_context"),
        patch("core.task_queue.workflow_executor.clear_task_context"),
    ):
        from core.task_queue.workflow_executor import run_task_workflow

        result = await run_task_workflow(
            task, queue_manager, checkpoint_mgr, budget_tracker
        )

    assert result["status"] == "deferred"

    # update_task called with DEFERRED status and next_run_after
    queue_manager.update_task.assert_called_once_with(
        task["id"],
        status=TaskStatus.DEFERRED.value,
        next_run_after=next_run,
        started_at=None,
    )

    # Checkpoint cleared (not failed)
    checkpoint_mgr.complete_work.assert_called_once_with(task["id"])
    checkpoint_mgr.fail_work.assert_not_called()


@pytest.mark.asyncio
async def test_waiting_status_does_not_mark_failed(mock_deps):
    """When workflow returns 'waiting', task is NOT marked as failed."""
    queue_manager, checkpoint_mgr, budget_tracker = mock_deps
    task = _make_task()
    workflow = _mock_workflow({"status": "waiting", "next_publish": "2026-03-01"})

    with (
        patch("core.task_queue.workflow_executor.get_workflow", return_value=workflow),
        patch("core.task_queue.workflow_executor.start_run"),
        patch("core.task_queue.workflow_executor.end_run"),
        patch("core.task_queue.workflow_executor.set_task_context"),
        patch("core.task_queue.workflow_executor.clear_task_context"),
    ):
        from core.task_queue.workflow_executor import run_task_workflow

        result = await run_task_workflow(
            task, queue_manager, checkpoint_mgr, budget_tracker
        )

    assert result["status"] == "waiting"

    # Should NOT be marked as failed
    queue_manager.mark_failed.assert_not_called()
    checkpoint_mgr.fail_work.assert_not_called()

    # Should NOT be marked as completed either
    queue_manager.mark_completed.assert_not_called()
