"""Tests for QueuePersistence, including v1-to-v2 migration.

Covers _migrate_v1_to_v2 partitioning logic: research vs publish task
routing, empty queue handling, and preservation of metadata fields.
"""


from core.task_queue.persistence import QueuePersistence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v1_research_task(
    task_id: str = "r1",
    task_type: str = "lit_review_full",
    category: str = "science",
) -> dict:
    """Build a minimal v1 research task dict."""
    return {
        "id": task_id,
        "task_type": task_type,
        "topic": "Test topic",
        "category": category,
        "priority": 2,
        "status": "pending",
        "quality": "standard",
        "created_at": "2026-01-15T10:00:00+00:00",
    }


def _make_v1_publish_task(
    task_id: str = "p1",
    task_type: str = "illustrate_and_publish",
    category: str = "science",
) -> dict:
    """Build a minimal v1 publish task dict."""
    return {
        "id": task_id,
        "task_type": task_type,
        "topic": "Publish topic",
        "category": category,
        "priority": 3,
        "status": "pending",
        "quality": "standard",
        "created_at": "2026-01-16T10:00:00+00:00",
        "source_task_id": "parent-id",
        "manifest_path": "/tmp/manifest.json",
        "items": [],
        "not_before": None,
    }


# ---------------------------------------------------------------------------
# _migrate_v1_to_v2
# ---------------------------------------------------------------------------


class TestMigrateV1ToV2:
    """Tests for QueuePersistence._migrate_v1_to_v2 static method."""

    def test_mixed_tasks_split_correctly(self):
        """v1 queue with both research and publish tasks partitions into correct arrays."""
        research = _make_v1_research_task(task_id="r1")
        web_research = _make_v1_research_task(task_id="r2", task_type="web_research")
        publish = _make_v1_publish_task(task_id="p1")

        v1_data = {
            "topics": [research, web_research, publish],
            "categories": ["science", "tech"],
            "last_category_index": 0,
            "last_updated": "2026-01-15T12:00:00+00:00",
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert result["version"] == "2.0"
        assert len(result["research_tasks"]) == 2
        assert len(result["publish_tasks"]) == 1

        research_ids = {t["id"] for t in result["research_tasks"]}
        assert research_ids == {"r1", "r2"}
        assert result["publish_tasks"][0]["id"] == "p1"

    def test_empty_topics_produces_empty_v2(self):
        """v1 queue with no tasks produces empty v2 research and publish arrays."""
        v1_data = {
            "topics": [],
            "categories": ["science"],
            "last_category_index": -1,
            "last_updated": "2026-01-15T12:00:00+00:00",
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert result["version"] == "2.0"
        assert result["research_tasks"] == []
        assert result["publish_tasks"] == []

    def test_publish_only_goes_to_publish_tasks(self):
        """v1 queue with only publish tasks puts everything in publish_tasks."""
        p1 = _make_v1_publish_task(task_id="p1")
        p2 = _make_v1_publish_task(task_id="p2")

        v1_data = {
            "topics": [p1, p2],
            "categories": ["science"],
            "last_category_index": -1,
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert result["research_tasks"] == []
        assert len(result["publish_tasks"]) == 2

    def test_research_only_goes_to_research_tasks(self):
        """v1 queue with only research tasks puts everything in research_tasks."""
        r1 = _make_v1_research_task(task_id="r1")
        r2 = _make_v1_research_task(task_id="r2", task_type="web_research")

        v1_data = {
            "topics": [r1, r2],
            "categories": ["science"],
            "last_category_index": 0,
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert len(result["research_tasks"]) == 2
        assert result["publish_tasks"] == []

    def test_preserves_categories_and_index(self):
        """Migration carries over categories and last_category_index."""
        v1_data = {
            "topics": [],
            "categories": ["alpha", "beta", "gamma"],
            "last_category_index": 2,
            "last_updated": "2026-01-15T12:00:00+00:00",
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert result["categories"] == ["alpha", "beta", "gamma"]
        assert result["last_category_index"] == 2

    def test_preserves_last_updated(self):
        """Migration carries over the original last_updated timestamp."""
        v1_data = {
            "topics": [],
            "categories": [],
            "last_category_index": -1,
            "last_updated": "2026-01-10T08:30:00+00:00",
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert result["last_updated"] == "2026-01-10T08:30:00+00:00"

    def test_missing_categories_defaults_to_empty(self):
        """If v1 data lacks categories, default to empty list."""
        v1_data = {
            "topics": [],
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert result["categories"] == []
        assert result["last_category_index"] == -1

    def test_task_without_task_type_defaults_to_research(self):
        """Tasks missing task_type field default to lit_review_full (research)."""
        task_no_type = {
            "id": "legacy",
            "topic": "Old task",
            "category": "science",
            "priority": 2,
            "status": "pending",
        }

        v1_data = {
            "topics": [task_no_type],
            "categories": ["science"],
            "last_category_index": -1,
        }

        result = QueuePersistence._migrate_v1_to_v2(v1_data)

        assert len(result["research_tasks"]) == 1
        assert result["publish_tasks"] == []
        assert result["research_tasks"][0]["id"] == "legacy"
