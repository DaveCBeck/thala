"""Tests for the illustrate_and_publish workflow."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.task_queue.workflows.illustrate_and_publish import IllustrateAndPublishWorkflow


def _make_manifest(tmp_path: Path, articles: list[dict] | None = None) -> Path:
    """Create a manifest.json and article files in tmp_path."""
    if articles is None:
        articles = [
            {"id": "overview", "title": "Overview Article", "filename": "overview.md"},
            {"id": "deep_dive_1", "title": "Deep Dive 1", "filename": "deep_dive_1.md"},
        ]

    for article in articles:
        (tmp_path / article["filename"]).write_text(f"# {article['title']}\n\nContent here.")

    manifest = {
        "topic": "Test Topic",
        "category": "science",
        "quality": "standard",
        "source_task_id": "parent-id",
        "output_dir": str(tmp_path),
        "articles": articles,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _make_task(tmp_path: Path, manifest_path: Path, num_articles: int = 2) -> dict:
    """Build an illustrate_and_publish task dict."""
    articles = [
        {"id": "overview", "title": "Overview Article", "filename": "overview.md"},
        {"id": "deep_dive_1", "title": "Deep Dive 1", "filename": "deep_dive_1.md"},
    ][:num_articles]

    items = [
        {
            "id": a["id"],
            "title": a["title"],
            "source_path": str(tmp_path / a["filename"]),
            "illustrated": False,
            "illustrated_path": None,
            "draft_id": None,
            "draft_url": None,
        }
        for a in articles
    ]

    return {
        "id": "test-task-id",
        "task_type": "illustrate_and_publish",
        "status": "in_progress",
        "category": "science",
        "priority": 2,
        "quality": "standard",
        "source_task_id": "parent-id",
        "topic": "Test Topic",
        "manifest_path": str(manifest_path),
        "items": items,
        "next_run_after": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "langsmith_run_id": None,
        "current_phase": None,
        "error_message": None,
        "notes": None,
        "tags": [],
    }


def _mock_illustrate_graph():
    """Mock illustrate_graph.ainvoke that returns illustrated content."""

    async def mock_ainvoke(input_dict):
        content = input_dict["input"]["markdown_document"]
        return {"illustrated_document": f"[ILLUSTRATED]\n{content}"}

    graph = MagicMock()
    graph.ainvoke = mock_ainvoke
    return graph


def _mock_daily_tracker(remaining: int = 10, acquire_results: list[bool] | None = None):
    """Create a mock daily tracker."""
    tracker = MagicMock()
    tracker.remaining = AsyncMock(return_value=remaining)
    if acquire_results is not None:
        tracker.try_acquire = AsyncMock(side_effect=acquire_results)
    else:
        tracker.try_acquire = AsyncMock(return_value=True)
    return tracker


@pytest.fixture
def workflow():
    return IllustrateAndPublishWorkflow()


class TestIllustrateAndPublishWorkflow:
    @pytest.mark.asyncio
    async def test_completes_when_all_articles_done(self, workflow, tmp_path):
        """Workflow returns 'success' when all articles are illustrated and published."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)
        tracker = _mock_daily_tracker(remaining=10)

        async def mock_publish(item, category):
            return {"post_id": f"draft-{item['id']}", "draft_url": f"https://example.com/{item['id']}"}

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
        ):
            workflow._publish_draft = mock_publish
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "success"
        for item in task["items"]:
            assert item["illustrated"] is True
            assert item["draft_id"] is not None

    @pytest.mark.asyncio
    async def test_defers_when_budget_exhausted(self, workflow, tmp_path):
        """Workflow returns 'deferred' when daily budget is 0 on fast-fail check."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)
        tracker = _mock_daily_tracker(remaining=0)

        with patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker):
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
            )

        assert result["status"] == "deferred"
        assert "next_run_after" in result

    @pytest.mark.asyncio
    async def test_defers_mid_loop_when_budget_runs_out(self, workflow, tmp_path):
        """Workflow defers after illustrating some articles when budget exhausted mid-loop."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)
        # remaining() is called per-article as a non-consuming check.
        # First article: remaining=1 (proceed), second article: remaining=0 (break).
        tracker = _mock_daily_tracker(remaining=1)
        tracker.remaining = AsyncMock(side_effect=[1, 1, 0])

        async def mock_publish(item, category):
            return {"post_id": f"draft-{item['id']}", "draft_url": f"https://example.com/{item['id']}"}

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
        ):
            workflow._publish_draft = mock_publish
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "deferred"
        # First article should be done
        assert task["items"][0]["illustrated"] is True
        assert task["items"][0]["draft_id"] is not None
        # Second article should not be done
        assert task["items"][1]["illustrated"] is False

    @pytest.mark.asyncio
    async def test_resumes_skipping_completed_articles(self, workflow, tmp_path):
        """On resume, already-completed articles are skipped."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)

        # First article already done from previous run
        (tmp_path / "overview_illustrated.md").write_text("# Already illustrated")
        task["items"][0]["illustrated"] = True
        task["items"][0]["illustrated_path"] = str(tmp_path / "overview_illustrated.md")
        task["items"][0]["draft_id"] = "existing-draft"
        task["items"][0]["draft_url"] = "https://example.com/existing"

        tracker = _mock_daily_tracker(remaining=10)

        async def mock_publish(item, category):
            return {"post_id": f"draft-{item['id']}", "draft_url": f"https://example.com/{item['id']}"}

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
        ):
            workflow._publish_draft = mock_publish
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "success"
        # First article kept its existing draft_id
        assert task["items"][0]["draft_id"] == "existing-draft"
        # Second article was done this run
        assert task["items"][1]["illustrated"] is True
        assert task["items"][1]["draft_id"] is not None

    @pytest.mark.asyncio
    async def test_fails_when_manifest_missing(self, workflow, tmp_path):
        """Workflow returns 'failed' when manifest file is missing."""
        task = _make_task(tmp_path, tmp_path / "nonexistent.json")

        result = await workflow.run(
            task,
            checkpoint_callback=MagicMock(),
            resume_from=None,
        )

        assert result["status"] == "failed"

    def test_properties(self, workflow):
        assert workflow.task_type == "illustrate_and_publish"
        assert workflow.is_zero_cost is False
        assert workflow.bypass_concurrency is True
        assert workflow.phases == ["processing", "complete"]
