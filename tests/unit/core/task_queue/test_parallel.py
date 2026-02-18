"""Tests for the parallel workflow supervisor.

Covers _select_tasks two-queue selection logic and run_parallel_tasks
top-level orchestration.
"""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.task_queue.parallel import (
    _select_publish_tasks,
    _select_research_tasks,
    _select_tasks,
    run_parallel_tasks,
)
from core.task_queue.schemas.enums import TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_research_task(
    task_id: str = "aaa",
    status: str = TaskStatus.PENDING.value,
    priority: int = 2,
    created_at: str | None = None,
    topic: str = "topic",
    category: str = "science",
) -> dict:
    """Build a minimal research task dict."""
    return {
        "id": task_id,
        "task_type": "lit_review_full",
        "category": category,
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


def _make_publish_task(
    task_id: str = "pub1",
    status: str = TaskStatus.PENDING.value,
    priority: int = 2,
    created_at: str | None = None,
    not_before: str | None = None,
    next_run_after: str | None = None,
    category: str = "science",
) -> dict:
    """Build a minimal publish task dict."""
    return {
        "id": task_id,
        "task_type": "illustrate_and_publish",
        "category": category,
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
        "topic": "test topic",
        "source_task_id": "parent-id",
        "manifest_path": "/tmp/manifest.json",
        "items": [],
        "not_before": not_before,
        "next_run_after": next_run_after,
    }


def _mock_queue_manager(
    research_tasks: list[dict] | None = None,
    publish_tasks: list[dict] | None = None,
) -> MagicMock:
    """Return a mock TaskQueueManager with two-queue structure."""
    queue = {
        "version": "2.0",
        "categories": ["science"],
        "last_category_index": -1,
        "research_tasks": research_tasks or [],
        "publish_tasks": publish_tasks or [],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

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
# _select_publish_tasks
# ---------------------------------------------------------------------------


class TestSelectPublishTasks:
    """Tests for _select_publish_tasks helper."""

    def test_selects_pending_without_not_before(self):
        """PENDING task with no not_before is immediately eligible."""
        task = _make_publish_task(task_id="p1")
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([task], count=5, now=now, resumable_ids=set())
        assert len(selected) == 1
        assert selected[0]["id"] == "p1"

    def test_selects_pending_with_past_not_before(self):
        """PENDING task with not_before in the past is eligible."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        task = _make_publish_task(task_id="p1", not_before=past)
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([task], count=5, now=now, resumable_ids=set())
        assert len(selected) == 1

    def test_skips_pending_with_future_not_before(self):
        """PENDING task with not_before in the future is invisible."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        task = _make_publish_task(task_id="p1", not_before=future)
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([task], count=5, now=now, resumable_ids=set())
        assert len(selected) == 0

    def test_selects_deferred_with_past_next_run(self):
        """DEFERRED task with next_run_after in the past is eligible."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        task = _make_publish_task(task_id="d1", status=TaskStatus.DEFERRED.value, next_run_after=past)
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([task], count=5, now=now, resumable_ids=set())
        assert len(selected) == 1

    def test_skips_deferred_with_future_next_run(self):
        """DEFERRED task with next_run_after in the future is skipped."""
        future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
        task = _make_publish_task(task_id="d1", status=TaskStatus.DEFERRED.value, next_run_after=future)
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([task], count=5, now=now, resumable_ids=set())
        assert len(selected) == 0

    def test_deferred_ignores_not_before(self):
        """DEFERRED task selection uses next_run_after, not not_before."""
        past_run = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        future_nb = (datetime.now(timezone.utc) + timedelta(hours=10)).isoformat()
        task = _make_publish_task(
            task_id="d1",
            status=TaskStatus.DEFERRED.value,
            not_before=future_nb,
            next_run_after=past_run,
        )
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([task], count=5, now=now, resumable_ids=set())
        assert len(selected) == 1

    def test_sorts_by_priority_desc(self):
        """Higher priority publish tasks are selected first."""
        low = _make_publish_task(task_id="low", priority=1)
        high = _make_publish_task(task_id="high", priority=4)
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([low, high], count=1, now=now, resumable_ids=set())
        assert selected[0]["id"] == "high"

    def test_resumable_first(self):
        """Resumable IN_PROGRESS publish tasks come before PENDING ones."""
        resumable = _make_publish_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value)
        pending = _make_publish_task(task_id="p1", priority=4)
        now = datetime.now(timezone.utc)
        selected = _select_publish_tasks([resumable, pending], count=1, now=now, resumable_ids={"r1"})
        assert selected[0]["id"] == "r1"


# ---------------------------------------------------------------------------
# _select_research_tasks
# ---------------------------------------------------------------------------


class TestSelectResearchTasks:
    """Tests for _select_research_tasks helper."""

    @patch(
        "core.task_queue.parallel.load_categories_from_publications",
        return_value=["alpha", "beta", "gamma"],
    )
    def test_category_rotation(self, _mock_cats):
        """Selects from different categories via round-robin."""
        t1 = _make_research_task(task_id="a1", category="alpha")
        t2 = _make_research_task(task_id="a2", category="alpha")
        t3 = _make_research_task(task_id="b1", category="beta")
        now = datetime.now(timezone.utc)

        selected, last_idx = _select_research_tasks(
            [t1, t2, t3],
            count=2,
            categories=["alpha", "beta", "gamma"],
            last_idx=-1,
            resumable_ids=set(),
            now=now,
        )

        categories = [t["category"] for t in selected]
        assert len(selected) == 2
        assert "alpha" in categories
        assert "beta" in categories

    def test_priority_ordering_within_category(self):
        """Higher priority tasks selected first within same category."""
        low = _make_research_task(task_id="low", priority=2, category="science")
        high = _make_research_task(task_id="high", priority=4, category="science")
        now = datetime.now(timezone.utc)

        selected, _ = _select_research_tasks(
            [low, high],
            count=1,
            categories=["science"],
            last_idx=-1,
            resumable_ids=set(),
            now=now,
        )

        assert selected[0]["id"] == "high"

    def test_fifo_within_same_priority(self):
        """Among same-priority tasks, earlier created_at comes first."""
        older = _make_research_task(task_id="older", priority=2, created_at="2025-01-01T00:00:00+00:00")
        newer = _make_research_task(task_id="newer", priority=2, created_at="2025-06-01T00:00:00+00:00")
        now = datetime.now(timezone.utc)

        selected, _ = _select_research_tasks(
            [newer, older],
            count=1,
            categories=["science"],
            last_idx=-1,
            resumable_ids=set(),
            now=now,
        )

        assert selected[0]["id"] == "older"

    def test_resumable_prioritized(self):
        """Resumable tasks come before higher-priority pending tasks."""
        resumable = _make_research_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value, priority=2)
        pending_high = _make_research_task(task_id="p1", priority=4)
        now = datetime.now(timezone.utc)

        selected, _ = _select_research_tasks(
            [resumable, pending_high],
            count=1,
            categories=["science"],
            last_idx=-1,
            resumable_ids={"r1"},
            now=now,
        )

        assert selected[0]["id"] == "r1"

    def test_deferred_with_past_next_run_eligible(self):
        """DEFERRED research task with expired next_run_after is eligible."""
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        deferred = _make_research_task(task_id="d1", status=TaskStatus.DEFERRED.value)
        deferred["next_run_after"] = past
        now = datetime.now(timezone.utc)

        selected, _ = _select_research_tasks(
            [deferred],
            count=5,
            categories=["science"],
            last_idx=-1,
            resumable_ids=set(),
            now=now,
        )

        assert len(selected) == 1
        assert selected[0]["id"] == "d1"

    def test_empty_list_returns_empty(self):
        """No candidates returns empty list."""
        now = datetime.now(timezone.utc)
        selected, last_idx = _select_research_tasks(
            [],
            count=5,
            categories=["science"],
            last_idx=-1,
            resumable_ids=set(),
            now=now,
        )
        assert selected == []


# ---------------------------------------------------------------------------
# _select_tasks (integrated)
# ---------------------------------------------------------------------------


class TestSelectTasks:
    """Tests for _select_tasks atomic selection."""

    def test_publish_tasks_selected_first(self):
        """Publish tasks fill slots before research tasks."""
        pub = _make_publish_task(task_id="pub1", priority=1)
        res = _make_research_task(task_id="res1", priority=4)
        qm = _mock_queue_manager(research_tasks=[res], publish_tasks=[pub])

        selected = _select_tasks(qm, count=1)

        assert len(selected) == 1
        assert selected[0]["id"] == "pub1"

    def test_research_fills_remaining_slots(self):
        """Research tasks fill slots not taken by publish tasks."""
        pub = _make_publish_task(task_id="pub1")
        res1 = _make_research_task(task_id="res1")
        res2 = _make_research_task(task_id="res2")
        qm = _mock_queue_manager(research_tasks=[res1, res2], publish_tasks=[pub])

        selected = _select_tasks(qm, count=3)

        assert len(selected) == 3
        selected_ids = {t["id"] for t in selected}
        assert "pub1" in selected_ids
        assert "res1" in selected_ids
        assert "res2" in selected_ids

    def test_marks_selected_as_in_progress(self):
        """Selected tasks get status=IN_PROGRESS and started_at set."""
        tasks = [_make_research_task(task_id="t1"), _make_research_task(task_id="t2")]
        qm = _mock_queue_manager(research_tasks=tasks)

        selected = _select_tasks(qm, count=2)

        for task in selected:
            assert task["status"] == TaskStatus.IN_PROGRESS.value
            assert task["started_at"] is not None

    def test_empty_queues_returns_empty(self):
        """Empty queues return empty list."""
        qm = _mock_queue_manager()

        selected = _select_tasks(qm, count=5)

        assert selected == []

    def test_resets_orphaned_in_progress(self):
        """Orphaned IN_PROGRESS tasks (no checkpoint) are reset to PENDING and selectable."""
        orphan = _make_research_task(task_id="orphan", status=TaskStatus.IN_PROGRESS.value)
        pending = _make_research_task(task_id="pending")
        qm = _mock_queue_manager(research_tasks=[orphan, pending])

        selected = _select_tasks(qm, count=10)

        assert len(selected) == 2
        selected_ids = {t["id"] for t in selected}
        assert "orphan" in selected_ids
        assert "pending" in selected_ids

    def test_writes_queue_after_selection(self):
        """write_queue is called to persist status changes."""
        tasks = [_make_research_task(task_id="t1")]
        qm = _mock_queue_manager(research_tasks=tasks)

        _select_tasks(qm, count=1)

        qm.persistence.write_queue.assert_called_once()

    def test_publish_with_future_not_before_invisible(self):
        """Publish tasks with not_before in future are not selected."""
        future = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        pub = _make_publish_task(task_id="pub1", not_before=future)
        res = _make_research_task(task_id="res1")
        qm = _mock_queue_manager(research_tasks=[res], publish_tasks=[pub])

        selected = _select_tasks(qm, count=5)

        assert len(selected) == 1
        assert selected[0]["id"] == "res1"

    def test_resets_unselected_resumable_to_pending(self):
        """When more resumable tasks than count, unselected ones are reset to PENDING."""
        r1 = _make_research_task(task_id="r1", status=TaskStatus.IN_PROGRESS.value, priority=3)
        r2 = _make_research_task(task_id="r2", status=TaskStatus.IN_PROGRESS.value, priority=2)
        r3 = _make_research_task(task_id="r3", status=TaskStatus.IN_PROGRESS.value, priority=1)
        qm = _mock_queue_manager(research_tasks=[r1, r2, r3])

        selected = _select_tasks(qm, count=2, resumable_ids={"r1", "r2", "r3"})

        assert len(selected) == 2
        selected_ids = {t["id"] for t in selected}

        # The unselected resumable task should be PENDING in written queue
        written_queue = qm.persistence.write_queue.call_args[0][0]
        for task in written_queue["research_tasks"]:
            if task["id"] not in selected_ids:
                assert task["status"] == TaskStatus.PENDING.value


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
        task = _make_research_task(task_id="budget-task")

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
            _make_research_task(task_id="t1"),
            _make_research_task(task_id="t2"),
        ]

        coordinator = MagicMock()
        coordinator.install_signal_handlers = MagicMock()
        coordinator.remove_signal_handlers = MagicMock()
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

            assert len(results) == 2
            assert results[0] == {"status": "completed"}
            assert isinstance(results[1], asyncio.CancelledError)

    @pytest.mark.asyncio
    async def test_workflow_exception_returned_not_raised(self):
        """When a workflow raises, the exception is returned (return_exceptions=True)."""
        task = _make_research_task(task_id="fail-task")

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
        task = _make_research_task(task_id="resume-task", status=TaskStatus.IN_PROGRESS.value)
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
