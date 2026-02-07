"""Tests for the parallel workflow supervisor.

Covers _select_tasks atomic selection logic and run_parallel_tasks
top-level orchestration.
"""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.task_queue.parallel import _select_tasks, run_parallel_tasks
from core.task_queue.schemas.enums import TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str = "aaa",
    status: str = TaskStatus.PENDING.value,
    priority: int = 2,
    created_at: str | None = None,
    topic: str = "topic",
) -> dict:
    """Build a minimal task dict matching the Task TypedDict shape."""
    return {
        "id": task_id,
        "task_type": "lit_review_full",
        "category": "science",
        "priority": priority,
        "status": status,
        "quality": "standard",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "langsmith_run_id": None,
        "current_phase": None,
        "error_message": None,
        "notes": None,
        "tags": [],
        "topic": topic,
        "research_questions": None,
        "language": "en",
        "date_range": None,
    }


def _mock_queue_manager(tasks: list[dict]) -> MagicMock:
    """Return a mock TaskQueueManager whose persistence holds *tasks*."""
    queue = {"topics": tasks}

    @contextmanager
    def _lock():
        yield

    persistence = MagicMock()
    persistence.lock = _lock
    persistence.read_queue.return_value = queue
    persistence.write_queue = MagicMock()

    qm = MagicMock()
    qm.persistence = persistence
    return qm


# ---------------------------------------------------------------------------
# _select_tasks
# ---------------------------------------------------------------------------

class TestSelectTasks:
    """Tests for _select_tasks atomic selection."""

    def test_selects_correct_count(self):
        """Given 10 pending tasks and count=3, selects exactly 3."""
        tasks = [_make_task(task_id=str(i)) for i in range(10)]
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=3)

        assert len(selected) == 3

    def test_marks_selected_as_in_progress(self):
        """Selected tasks get status=IN_PROGRESS and started_at set."""
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=2)

        for task in selected:
            assert task["status"] == TaskStatus.IN_PROGRESS.value
            assert task["started_at"] is not None

    def test_empty_queue_returns_empty(self):
        """Given 0 pending tasks, returns empty list."""
        qm = _mock_queue_manager([])

        selected = _select_tasks(qm, count=5)

        assert selected == []

    def test_fewer_tasks_than_count(self):
        """Given 2 pending tasks and count=5, selects 2."""
        tasks = [_make_task(task_id="t1"), _make_task(task_id="t2")]
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 2

    def test_priority_ordering(self):
        """Higher priority tasks are selected first (4 before 2)."""
        low = _make_task(task_id="low", priority=2, created_at="2025-01-01T00:00:00+00:00")
        high = _make_task(task_id="high", priority=4, created_at="2025-01-02T00:00:00+00:00")
        tasks = [low, high]  # low comes first in the list
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=1)

        assert selected[0]["id"] == "high"

    def test_fifo_within_same_priority(self):
        """Among same-priority tasks, earlier created_at comes first."""
        older = _make_task(task_id="older", priority=2, created_at="2025-01-01T00:00:00+00:00")
        newer = _make_task(task_id="newer", priority=2, created_at="2025-06-01T00:00:00+00:00")
        tasks = [newer, older]  # newer comes first in the list
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=1)

        assert selected[0]["id"] == "older"

    def test_only_selects_pending(self):
        """Ignores tasks already IN_PROGRESS or COMPLETED."""
        pending = _make_task(task_id="pending", status=TaskStatus.PENDING.value)
        in_progress = _make_task(task_id="running", status=TaskStatus.IN_PROGRESS.value)
        completed = _make_task(task_id="done", status=TaskStatus.COMPLETED.value)
        tasks = [pending, in_progress, completed]
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=10)

        assert len(selected) == 1
        assert selected[0]["id"] == "pending"

    def test_writes_queue_after_selection(self):
        """write_queue is called to persist the status changes."""
        tasks = [_make_task(task_id="t1")]
        qm = _mock_queue_manager(tasks)

        _select_tasks(qm, count=1)

        qm.persistence.write_queue.assert_called_once()

    def test_does_not_mutate_non_selected_tasks(self):
        """Tasks not selected keep their original PENDING status."""
        tasks = [
            _make_task(task_id="t1", priority=4),
            _make_task(task_id="t2", priority=1),
        ]
        qm = _mock_queue_manager(tasks)

        _select_tasks(qm, count=1)

        # t2 (lower priority, not selected) should remain PENDING
        t2 = next(t for t in tasks if t["id"] == "t2")
        assert t2["status"] == TaskStatus.PENDING.value
        assert t2["started_at"] is None


# ---------------------------------------------------------------------------
# run_parallel_tasks
# ---------------------------------------------------------------------------

class TestRunParallelTasks:
    """Tests for run_parallel_tasks top-level orchestration."""

    @pytest.mark.asyncio
    async def test_empty_queue_returns_empty_list(self):
        """When no pending tasks, returns []."""
        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator") as mock_coord_fn,
            patch("core.task_queue.parallel.TaskQueueManager") as mock_qm_cls,
            patch("core.task_queue.parallel.CheckpointManager"),
            patch("core.task_queue.parallel.BudgetTracker"),
            patch("core.task_queue.parallel.asyncio.to_thread", return_value=[]),
        ):
            coordinator = MagicMock()
            coordinator.install_signal_handlers = MagicMock()
            coordinator.remove_signal_handlers = MagicMock()
            mock_coord_fn.return_value = coordinator

            result = await run_parallel_tasks(count=5)

            assert result == []

    @pytest.mark.asyncio
    async def test_budget_check_skips_tasks(self):
        """When budget_tracker.should_proceed() returns False, task is skipped."""
        task = _make_task(task_id="budget-task")

        coordinator = MagicMock()
        coordinator.install_signal_handlers = MagicMock()
        coordinator.remove_signal_handlers = MagicMock()
        coordinator.wait_or_shutdown = AsyncMock(return_value=False)

        budget = MagicMock()
        budget.should_proceed.return_value = (False, "over limit")

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager"),
            patch("core.task_queue.parallel.BudgetTracker", return_value=budget),
            patch("core.task_queue.parallel.asyncio.to_thread", return_value=[task]),
            patch("core.task_queue.parallel.cleanup_supervisor_resources", new_callable=AsyncMock),
        ):
            results = await run_parallel_tasks(count=1, stagger_minutes=0)

            assert len(results) == 1
            assert results[0]["status"] == "skipped"
            assert "budget" in results[0]["reason"]

    @pytest.mark.asyncio
    async def test_shutdown_during_stagger_skips_remaining(self):
        """When coordinator signals shutdown during stagger wait, CancelledError is returned."""
        tasks = [
            _make_task(task_id="t1"),
            _make_task(task_id="t2"),
        ]

        coordinator = MagicMock()
        coordinator.install_signal_handlers = MagicMock()
        coordinator.remove_signal_handlers = MagicMock()
        # First task (index=0) has no stagger; second task (index=1) gets shutdown
        coordinator.wait_or_shutdown = AsyncMock(return_value=True)

        budget = MagicMock()
        budget.should_proceed.return_value = (True, "")

        mock_run_workflow = AsyncMock(return_value={"status": "completed"})

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager"),
            patch("core.task_queue.parallel.BudgetTracker", return_value=budget),
            patch("core.task_queue.parallel.asyncio.to_thread", return_value=tasks),
            patch("core.task_queue.parallel.run_task_workflow", mock_run_workflow),
            patch("core.task_queue.parallel.cleanup_supervisor_resources", new_callable=AsyncMock),
        ):
            results = await run_parallel_tasks(count=2, stagger_minutes=5)

            # First task (index=0) runs normally, second (index=1) gets CancelledError
            assert len(results) == 2
            assert results[0] == {"status": "completed"}
            assert isinstance(results[1], asyncio.CancelledError)

    @pytest.mark.asyncio
    async def test_workflow_exception_returned_not_raised(self):
        """When a workflow raises, the exception is returned (return_exceptions=True)."""
        task = _make_task(task_id="fail-task")

        coordinator = MagicMock()
        coordinator.install_signal_handlers = MagicMock()
        coordinator.remove_signal_handlers = MagicMock()
        coordinator.wait_or_shutdown = AsyncMock(return_value=False)

        budget = MagicMock()
        budget.should_proceed.return_value = (True, "")

        mock_run_workflow = AsyncMock(side_effect=RuntimeError("workflow boom"))

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager"),
            patch("core.task_queue.parallel.BudgetTracker", return_value=budget),
            patch("core.task_queue.parallel.asyncio.to_thread", return_value=[task]),
            patch("core.task_queue.parallel.run_task_workflow", mock_run_workflow),
            patch("core.task_queue.parallel.cleanup_supervisor_resources", new_callable=AsyncMock),
        ):
            results = await run_parallel_tasks(count=1, stagger_minutes=0)

            assert len(results) == 1
            assert isinstance(results[0], RuntimeError)
            assert "workflow boom" in str(results[0])
