"""Tests for task_selector._find_bypass_task."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from core.task_queue.schemas.enums import TaskStatus
from core.task_queue.task_selector import _find_bypass_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "aaa",
    status: str = TaskStatus.PENDING.value,
    task_type: str = "publish_series",
    next_run_after: str | None = None,
) -> dict:
    """Minimal task dict for selector tests."""
    t = {
        "id": task_id,
        "task_type": task_type,
        "status": status,
    }
    if next_run_after is not None:
        t["next_run_after"] = next_run_after
    return t


def _mock_qm(pending: list[dict] | None = None, deferred: list[dict] | None = None) -> MagicMock:
    """Return a mock TaskQueueManager with canned list_tasks results."""
    qm = MagicMock()

    def _list_tasks(status=None):
        if status == TaskStatus.PENDING:
            return pending or []
        if status == TaskStatus.DEFERRED:
            return deferred or []
        return []

    qm.list_tasks.side_effect = _list_tasks
    return qm


class _BypassWorkflow:
    bypass_concurrency = True


class _NormalWorkflow:
    bypass_concurrency = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindBypassTask:
    """_find_bypass_task selection logic."""

    @patch("core.task_queue.task_selector.get_workflow", return_value=_BypassWorkflow())
    def test_returns_pending_bypass_task(self, _mock_wf):
        task = _make_task(status=TaskStatus.PENDING.value)
        qm = _mock_qm(pending=[task])
        assert _find_bypass_task(qm) is task

    @patch("core.task_queue.task_selector.get_workflow", return_value=_NormalWorkflow())
    def test_returns_none_when_no_bypass(self, _mock_wf):
        task = _make_task(status=TaskStatus.PENDING.value, task_type="lit_review_full")
        qm = _mock_qm(pending=[task])
        assert _find_bypass_task(qm) is None

    @patch("core.task_queue.task_selector.get_workflow", return_value=_BypassWorkflow())
    def test_deferred_past_next_run_after_is_eligible(self, _mock_wf):
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        task = _make_task(status=TaskStatus.DEFERRED.value, next_run_after=past)
        qm = _mock_qm(deferred=[task])
        assert _find_bypass_task(qm) is task

    @patch("core.task_queue.task_selector.get_workflow", return_value=_BypassWorkflow())
    def test_deferred_missing_next_run_after_is_eligible(self, _mock_wf):
        task = _make_task(status=TaskStatus.DEFERRED.value)
        qm = _mock_qm(deferred=[task])
        assert _find_bypass_task(qm) is task

    @patch("core.task_queue.task_selector.get_workflow", return_value=_BypassWorkflow())
    def test_deferred_future_next_run_after_is_skipped(self, _mock_wf):
        """DEFERRED tasks with future next_run_after must NOT be picked up."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        task = _make_task(status=TaskStatus.DEFERRED.value, next_run_after=future)
        qm = _mock_qm(deferred=[task])
        assert _find_bypass_task(qm) is None

    @patch("core.task_queue.task_selector.get_workflow", return_value=_BypassWorkflow())
    def test_deferred_malformed_next_run_after_treated_as_eligible(self, _mock_wf):
        task = _make_task(status=TaskStatus.DEFERRED.value, next_run_after="not-a-date")
        qm = _mock_qm(deferred=[task])
        assert _find_bypass_task(qm) is task
