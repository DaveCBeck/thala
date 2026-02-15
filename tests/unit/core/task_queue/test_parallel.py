"""Tests for the parallel workflow supervisor.

Covers _select_tasks atomic selection logic and run_parallel_tasks
top-level orchestration.
"""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
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

    def test_resets_orphaned_in_progress_and_selects_them(self):
        """Orphaned IN_PROGRESS tasks (no checkpoint) are reset to PENDING and become selectable."""
        pending = _make_task(task_id="pending", status=TaskStatus.PENDING.value)
        in_progress = _make_task(task_id="running", status=TaskStatus.IN_PROGRESS.value)
        completed = _make_task(task_id="done", status=TaskStatus.COMPLETED.value)
        tasks = [pending, in_progress, completed]
        qm = _mock_queue_manager(tasks)

        selected = _select_tasks(qm, count=10)

        # Orphaned in_progress was reset to PENDING, so both are selectable
        assert len(selected) == 2
        selected_ids = {t["id"] for t in selected}
        assert "pending" in selected_ids
        assert "running" in selected_ids
        # COMPLETED task is still excluded
        assert "done" not in selected_ids

    def test_writes_queue_after_selection(self):
        """write_queue is called to persist the status changes."""
        tasks = [_make_task(task_id="t1")]
        qm = _mock_queue_manager(tasks)

        _select_tasks(qm, count=1)

        qm.persistence.write_queue.assert_called_once()

    @patch(
        "core.task_queue.parallel.load_categories_from_publications",
        return_value=["alpha", "beta", "gamma"],
    )
    def test_category_rotation_selects_from_different_categories(self, _mock_cats):
        """Selecting 2 tasks should pick from 2 different categories, not the same one twice."""
        t1 = _make_task(task_id="a1", topic="Alpha topic 1")
        t1["category"] = "alpha"
        t2 = _make_task(task_id="a2", topic="Alpha topic 2")
        t2["category"] = "alpha"
        t3 = _make_task(task_id="b1", topic="Beta topic 1")
        t3["category"] = "beta"

        qm = _mock_queue_manager([t1, t2, t3])

        selected = _select_tasks(qm, count=2)

        categories = [t["category"] for t in selected]
        assert len(selected) == 2
        assert "alpha" in categories
        assert "beta" in categories

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
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.get_incomplete_work = AsyncMock(return_value=[])

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator") as mock_coord_fn,
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager", return_value=checkpoint_mgr),
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

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.get_incomplete_work = AsyncMock(return_value=[])

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager", return_value=checkpoint_mgr),
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

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.get_incomplete_work = AsyncMock(return_value=[])

        mock_run_workflow = AsyncMock(return_value={"status": "completed"})

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager", return_value=checkpoint_mgr),
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

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.get_incomplete_work = AsyncMock(return_value=[])

        mock_run_workflow = AsyncMock(side_effect=RuntimeError("workflow boom"))

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager", return_value=checkpoint_mgr),
            patch("core.task_queue.parallel.BudgetTracker", return_value=budget),
            patch("core.task_queue.parallel.asyncio.to_thread", return_value=[task]),
            patch("core.task_queue.parallel.run_task_workflow", mock_run_workflow),
            patch("core.task_queue.parallel.cleanup_supervisor_resources", new_callable=AsyncMock),
        ):
            results = await run_parallel_tasks(count=1, stagger_minutes=0)

            assert len(results) == 1
            assert isinstance(results[0], RuntimeError)
            assert "workflow boom" in str(results[0])

    @pytest.mark.asyncio
    async def test_passes_resume_from_for_checkpointed_tasks(self):
        """Resumable tasks pass their checkpoint as resume_from to run_task_workflow."""
        task = _make_task(task_id="resume-task", status=TaskStatus.IN_PROGRESS.value)
        checkpoint = {"task_id": "resume-task", "phase": "supervision", "phase_outputs": {}}

        coordinator = MagicMock()
        coordinator.install_signal_handlers = MagicMock()
        coordinator.remove_signal_handlers = MagicMock()
        coordinator.wait_or_shutdown = AsyncMock(return_value=False)

        budget = MagicMock()
        budget.should_proceed.return_value = (True, "")

        checkpoint_mgr = MagicMock()
        checkpoint_mgr.get_incomplete_work = AsyncMock(return_value=[checkpoint])

        mock_run_workflow = AsyncMock(return_value={"status": "completed"})

        with (
            patch("core.task_queue.parallel.get_shutdown_coordinator", return_value=coordinator),
            patch("core.task_queue.parallel.TaskQueueManager"),
            patch("core.task_queue.parallel.CheckpointManager", return_value=checkpoint_mgr),
            patch("core.task_queue.parallel.BudgetTracker", return_value=budget),
            patch("core.task_queue.parallel.asyncio.to_thread", return_value=[task]),
            patch("core.task_queue.parallel.run_task_workflow", mock_run_workflow),
            patch("core.task_queue.parallel.cleanup_supervisor_resources", new_callable=AsyncMock),
        ):
            await run_parallel_tasks(count=1, stagger_minutes=0)

            mock_run_workflow.assert_called_once()
            call_kwargs = mock_run_workflow.call_args
            assert call_kwargs.kwargs.get("resume_from") == checkpoint


# ---------------------------------------------------------------------------
# _select_tasks — checkpoint-aware behaviour
# ---------------------------------------------------------------------------

class TestSelectTasksCheckpointAware:
    """Tests for _select_tasks resumable/orphan handling."""

    def test_prioritizes_resumable_over_pending(self):
        """Resumable IN_PROGRESS tasks are selected before PENDING tasks."""
        resumable = _make_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value, priority=2)
        pending_high = _make_task(task_id="p1", status=TaskStatus.PENDING.value, priority=4)
        pending_low = _make_task(task_id="p2", status=TaskStatus.PENDING.value, priority=2)
        qm = _mock_queue_manager([resumable, pending_high, pending_low])

        selected = _select_tasks(qm, count=1, resumable_ids={"r1"})

        assert len(selected) == 1
        assert selected[0]["id"] == "r1"

    def test_resets_orphaned_in_progress_without_checkpoint(self):
        """IN_PROGRESS tasks without checkpoint IDs are reset to PENDING."""
        orphan = _make_task(task_id="orphan", status=TaskStatus.IN_PROGRESS.value)
        orphan["started_at"] = "2025-01-01T00:00:00+00:00"
        resumable = _make_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value)
        qm = _mock_queue_manager([orphan, resumable])

        _select_tasks(qm, count=2, resumable_ids={"r1"})

        # Check the written queue state
        written_queue = qm.persistence.write_queue.call_args[0][0]
        orphan_task = next(t for t in written_queue["topics"] if t["id"] == "orphan")
        # Orphan was reset then re-selected as PENDING → marked IN_PROGRESS
        assert orphan_task["status"] == TaskStatus.IN_PROGRESS.value

    def test_resets_unselected_resumable_to_pending(self):
        """When more resumable tasks than count, unselected ones are reset to PENDING."""
        r1 = _make_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value, priority=3)
        r2 = _make_task(task_id="r2", status=TaskStatus.IN_PROGRESS.value, priority=2)
        r3 = _make_task(task_id="r3", status=TaskStatus.IN_PROGRESS.value, priority=1)
        qm = _mock_queue_manager([r1, r2, r3])

        selected = _select_tasks(qm, count=2, resumable_ids={"r1", "r2", "r3"})

        assert len(selected) == 2
        selected_ids = {t["id"] for t in selected}

        # The unselected resumable task should be PENDING in written queue
        written_queue = qm.persistence.write_queue.call_args[0][0]
        for task in written_queue["topics"]:
            if task["id"] not in selected_ids:
                assert task["status"] == TaskStatus.PENDING.value

    def test_fills_remaining_slots_from_pending(self):
        """After picking resumable tasks, fills remaining slots from PENDING pool."""
        resumable = _make_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value)
        p1 = _make_task(task_id="p1", status=TaskStatus.PENDING.value, priority=3)
        p2 = _make_task(task_id="p2", status=TaskStatus.PENDING.value, priority=2)
        qm = _mock_queue_manager([resumable, p1, p2])

        selected = _select_tasks(qm, count=3, resumable_ids={"r1"})

        assert len(selected) == 3
        selected_ids = {t["id"] for t in selected}
        assert "r1" in selected_ids
        assert "p1" in selected_ids
        assert "p2" in selected_ids

    def test_no_resumable_ids_behaves_like_before(self):
        """Without resumable_ids, all IN_PROGRESS tasks are reset and only PENDING selected."""
        in_progress = _make_task(task_id="ip", status=TaskStatus.IN_PROGRESS.value)
        pending = _make_task(task_id="p1", status=TaskStatus.PENDING.value)
        qm = _mock_queue_manager([in_progress, pending])

        selected = _select_tasks(qm, count=5)

        # Both should be selected (orphan reset to PENDING, then selected)
        assert len(selected) == 2

    @patch(
        "core.task_queue.parallel.load_categories_from_publications",
        return_value=["alpha", "beta"],
    )
    def test_resumable_uses_category_rotation(self, _mock_cats):
        """Resumable tasks from different categories are selected via round-robin."""
        r1 = _make_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value)
        r1["category"] = "alpha"
        r2 = _make_task(task_id="r2", status=TaskStatus.IN_PROGRESS.value)
        r2["category"] = "beta"
        r3 = _make_task(task_id="r3", status=TaskStatus.IN_PROGRESS.value)
        r3["category"] = "alpha"
        qm = _mock_queue_manager([r1, r2, r3])

        selected = _select_tasks(qm, count=2, resumable_ids={"r1", "r2", "r3"})

        categories = [t["category"] for t in selected]
        assert "alpha" in categories
        assert "beta" in categories


# ---------------------------------------------------------------------------
# _select_tasks — DEFERRED task handling
# ---------------------------------------------------------------------------

class TestSelectTasksDeferred:
    """Tests for DEFERRED task selection in _select_tasks."""

    def test_selects_deferred_with_expired_next_run(self):
        """DEFERRED tasks whose next_run_after has passed are included in candidate pool."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        deferred = _make_task(task_id="d1", status=TaskStatus.DEFERRED.value)
        deferred["next_run_after"] = past
        qm = _mock_queue_manager([deferred])

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 1
        assert selected[0]["id"] == "d1"
        assert selected[0]["status"] == TaskStatus.IN_PROGRESS.value

    def test_skips_deferred_with_future_next_run(self):
        """DEFERRED tasks whose next_run_after is in the future are skipped."""
        future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
        deferred = _make_task(task_id="d1", status=TaskStatus.DEFERRED.value)
        deferred["next_run_after"] = future
        pending = _make_task(task_id="p1")
        qm = _mock_queue_manager([deferred, pending])

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 1
        assert selected[0]["id"] == "p1"

    def test_deferred_without_next_run_treated_as_eligible(self):
        """DEFERRED tasks missing next_run_after are treated as immediately eligible."""
        deferred = _make_task(task_id="d1", status=TaskStatus.DEFERRED.value)
        # No next_run_after key at all
        qm = _mock_queue_manager([deferred])

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 1
        assert selected[0]["id"] == "d1"

    def test_deferred_with_malformed_next_run_treated_as_eligible(self):
        """DEFERRED tasks with invalid next_run_after are treated as eligible."""
        deferred = _make_task(task_id="d1", status=TaskStatus.DEFERRED.value)
        deferred["next_run_after"] = "not-a-date"
        qm = _mock_queue_manager([deferred])

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 1

    def test_deferred_mixed_with_pending(self):
        """Both DEFERRED (eligible) and PENDING tasks can be selected together."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        deferred = _make_task(task_id="d1", status=TaskStatus.DEFERRED.value)
        deferred["next_run_after"] = past
        pending = _make_task(task_id="p1")
        qm = _mock_queue_manager([deferred, pending])

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 2
        selected_ids = {t["id"] for t in selected}
        assert "d1" in selected_ids
        assert "p1" in selected_ids
