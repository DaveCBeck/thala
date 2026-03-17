"""Tests for TaskQueueManager.update_task() field allowlisting.

Verifies that protected fields (id, task_type, created_at) cannot be
overwritten and arbitrary keys cannot be injected via update_task().
"""

import json

import pytest

from core.task_queue.queue_manager import TaskQueueManager, _MUTABLE_TASK_FIELDS


@pytest.fixture
def queue_manager(tmp_path):
    """Create a TaskQueueManager with an isolated temp directory."""
    # Provide a minimal publications.json so _ensure_queue_exists works
    publications = tmp_path / "publications.json"
    publications.write_text(json.dumps({"publications": [{"name": "test", "categories": ["science"]}]}))

    # Monkey-patch PUBLICATIONS_FILE for this test
    import core.task_queue.queue_manager as qm_module
    original = qm_module.PUBLICATIONS_FILE
    qm_module.PUBLICATIONS_FILE = publications
    try:
        mgr = TaskQueueManager(queue_dir=tmp_path)
        yield mgr
    finally:
        qm_module.PUBLICATIONS_FILE = original


@pytest.fixture
def task_id(queue_manager):
    """Add a task and return its ID."""
    return queue_manager.add_task(
        category="science",
        topic="Test topic for allowlist",
        task_type="lit_review_full",
    )


class TestUpdateTaskAllowlist:
    """update_task() only applies changes for fields in _MUTABLE_TASK_FIELDS."""

    def test_mutable_field_is_applied(self, queue_manager, task_id):
        """Allowed fields (e.g. status, notes) are written through."""
        result = queue_manager.update_task(task_id, status="completed", notes="updated")
        assert result is True

        task = queue_manager.get_task(task_id)
        assert task["status"] == "completed"
        assert task["notes"] == "updated"

    def test_protected_id_cannot_be_overwritten(self, queue_manager, task_id):
        """The 'id' field must not change via update_task()."""
        result = queue_manager.update_task(task_id, id="evil-id")
        # All keys were filtered out -> returns False
        assert result is False

        task = queue_manager.get_task(task_id)
        assert task["id"] == task_id

    def test_protected_task_type_cannot_be_overwritten(self, queue_manager, task_id):
        """The 'task_type' field must not change via update_task()."""
        result = queue_manager.update_task(task_id, task_type="evil_type")
        assert result is False

        task = queue_manager.get_task(task_id)
        assert task["task_type"] == "lit_review_full"

    def test_protected_created_at_cannot_be_overwritten(self, queue_manager, task_id):
        """The 'created_at' field must not change via update_task()."""
        task_before = queue_manager.get_task(task_id)
        original_created_at = task_before["created_at"]

        result = queue_manager.update_task(task_id, created_at="1999-01-01T00:00:00+00:00")
        assert result is False

        task_after = queue_manager.get_task(task_id)
        assert task_after["created_at"] == original_created_at

    def test_arbitrary_keys_are_not_injected(self, queue_manager, task_id):
        """Keys not in any task schema are silently dropped."""
        result = queue_manager.update_task(task_id, evil_key="payload", __class__="hack")
        assert result is False

        task = queue_manager.get_task(task_id)
        assert "evil_key" not in task
        assert "__class__" not in task

    def test_mixed_allowed_and_blocked_fields(self, queue_manager, task_id):
        """When both allowed and blocked fields are passed, only allowed ones apply."""
        result = queue_manager.update_task(
            task_id,
            status="failed",
            error_message="something broke",
            id="evil-id",
            created_at="1999-01-01T00:00:00+00:00",
            evil_key="payload",
        )
        assert result is True

        task = queue_manager.get_task(task_id)
        # Allowed fields applied
        assert task["status"] == "failed"
        assert task["error_message"] == "something broke"
        # Protected fields unchanged
        assert task["id"] == task_id
        assert task["task_type"] == "lit_review_full"
        assert task["created_at"] != "1999-01-01T00:00:00+00:00"
        # Arbitrary key not present
        assert "evil_key" not in task

    def test_items_field_is_mutable(self, queue_manager, task_id):
        """The 'items' field (used by illustrate_and_export) is in the allowlist."""
        items = [{"id": "overview", "title": "Test", "illustrated": True}]
        result = queue_manager.update_task(task_id, items=items)
        assert result is True

        task = queue_manager.get_task(task_id)
        assert task["items"] == items

    def test_deferred_fields_are_mutable(self, queue_manager, task_id):
        """Fields used by the DEFERRED lifecycle are in the allowlist."""
        result = queue_manager.update_task(
            task_id,
            status="deferred",
            next_run_after="2026-02-20T00:00:00+00:00",
            started_at=None,
        )
        assert result is True

        task = queue_manager.get_task(task_id)
        assert task["status"] == "deferred"
        assert task["next_run_after"] == "2026-02-20T00:00:00+00:00"
        assert task["started_at"] is None

    def test_nonexistent_task_returns_false(self, queue_manager):
        """Updating a task that doesn't exist returns False."""
        result = queue_manager.update_task("nonexistent-id", status="completed")
        assert result is False

    def test_only_blocked_fields_returns_false_without_disk_write(self, queue_manager, task_id):
        """When all fields are filtered out, no disk I/O occurs (returns False early)."""
        # This tests the early-return optimization: if filtered dict is empty,
        # we don't even acquire the lock or read the queue file.
        result = queue_manager.update_task(task_id, id="evil", task_type="evil", created_at="evil")
        assert result is False


class TestMutableFieldsCoverage:
    """Verify that _MUTABLE_TASK_FIELDS covers all legitimate fields from schemas."""

    def test_protected_fields_not_in_allowlist(self):
        """id, task_type, created_at must never appear in the mutable set."""
        assert "id" not in _MUTABLE_TASK_FIELDS
        assert "task_type" not in _MUTABLE_TASK_FIELDS
        assert "created_at" not in _MUTABLE_TASK_FIELDS

    def test_lifecycle_fields_in_allowlist(self):
        """Core lifecycle fields used by mark_* methods are allowed."""
        for field in ("status", "started_at", "completed_at", "error_message",
                      "current_phase", "langsmith_run_id"):
            assert field in _MUTABLE_TASK_FIELDS, f"{field} missing from allowlist"

    def test_scheduling_fields_in_allowlist(self):
        """Scheduling-related fields are allowed."""
        for field in ("priority", "quality", "category", "next_run_after", "not_before"):
            assert field in _MUTABLE_TASK_FIELDS, f"{field} missing from allowlist"

    def test_content_fields_in_allowlist(self):
        """Content fields editable before execution are allowed."""
        for field in ("topic", "query", "research_questions", "language", "date_range"):
            assert field in _MUTABLE_TASK_FIELDS, f"{field} missing from allowlist"

    def test_metadata_fields_in_allowlist(self):
        """Metadata fields are allowed."""
        for field in ("notes", "tags"):
            assert field in _MUTABLE_TASK_FIELDS, f"{field} missing from allowlist"

    def test_publish_task_fields_in_allowlist(self):
        """Publish-task-specific fields are allowed."""
        for field in ("items", "source_task_id", "manifest_path"):
            assert field in _MUTABLE_TASK_FIELDS, f"{field} missing from allowlist"
