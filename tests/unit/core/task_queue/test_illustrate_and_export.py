"""Tests for the illustrate_and_export workflow."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.task_queue.workflows.illustrate_and_export import IllustrateAndExportWorkflow


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
    """Build an illustrate_and_export task dict."""
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
            "exported": False,
        }
        for a in articles
    ]

    return {
        "id": "test-task-id",
        "task_type": "illustrate_and_export",
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
    """Mock illustrate_graph.ainvoke that returns illustrated content + visual_identity."""
    from workflows.output.illustrate.schemas import VisualIdentity

    vi = VisualIdentity(
        primary_style="editorial watercolor",
        color_palette=["warm amber", "deep teal"],
        mood="contemplative",
        lighting="soft diffused",
        avoid=["neon colors"],
    )

    async def mock_ainvoke(input_dict):
        content = input_dict["input"]["markdown_document"]
        return {"illustrated_document": f"[ILLUSTRATED]\n{content}", "visual_identity": vi}

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
    return IllustrateAndExportWorkflow()


class TestIllustrateAndExportWorkflow:
    @pytest.mark.asyncio
    async def test_completes_when_all_articles_illustrated(self, workflow, tmp_path):
        """Workflow returns 'success' when all articles are illustrated and exported."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)
        tracker = _mock_daily_tracker(remaining=10)

        mock_export = MagicMock(return_value=tmp_path / "export" / "batch_0001")
        mock_rsync = AsyncMock(return_value=True)

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
            patch("core.task_queue.workflows.illustrate_and_export.export_batch", mock_export),
            patch("core.task_queue.workflows.illustrate_and_export.rsync_batch", mock_rsync),
        ):
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
            assert item["exported"] is True
        mock_export.assert_called_once()
        mock_rsync.assert_called_once()

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

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
        ):
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "deferred"
        # First article should be illustrated
        assert task["items"][0]["illustrated"] is True
        # Second article should not be done
        assert task["items"][1]["illustrated"] is False

    @pytest.mark.asyncio
    async def test_resumes_skipping_illustrated_articles(self, workflow, tmp_path):
        """On resume, already-illustrated articles are skipped."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)

        # First article already done from previous run
        (tmp_path / "overview_illustrated.md").write_text("# Already illustrated")
        task["items"][0]["illustrated"] = True
        task["items"][0]["illustrated_path"] = str(tmp_path / "overview_illustrated.md")

        tracker = _mock_daily_tracker(remaining=10)

        mock_export = MagicMock(return_value=tmp_path / "export" / "batch_0001")
        mock_rsync = AsyncMock(return_value=True)

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
            patch("core.task_queue.workflows.illustrate_and_export.export_batch", mock_export),
            patch("core.task_queue.workflows.illustrate_and_export.rsync_batch", mock_rsync),
        ):
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "success"
        # Both articles illustrated
        assert task["items"][0]["illustrated"] is True
        assert task["items"][1]["illustrated"] is True

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

    @pytest.mark.asyncio
    async def test_defers_when_rsync_fails(self, workflow, tmp_path):
        """Workflow defers when rsync to VPS fails."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)
        tracker = _mock_daily_tracker(remaining=10)

        mock_export = MagicMock(return_value=tmp_path / "export" / "batch_0001")
        mock_rsync = AsyncMock(return_value=False)

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
            patch("core.task_queue.workflows.illustrate_and_export.export_batch", mock_export),
            patch("core.task_queue.workflows.illustrate_and_export.rsync_batch", mock_rsync),
        ):
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "deferred"
        assert "next_run_after" in result

    @pytest.mark.asyncio
    async def test_caches_vi_from_first_article(self, workflow, tmp_path):
        """First article has no VI override; second article gets cached VI."""
        manifest_path = _make_manifest(tmp_path)
        task = _make_task(tmp_path, manifest_path)
        tracker = _mock_daily_tracker(remaining=10)

        # Track configs passed to _illustrate_article
        configs_seen = []
        original_illustrate = workflow._illustrate_article.__func__

        async def spy_illustrate(self_, item, output_dir, graph, config=None):
            configs_seen.append(config)
            return await original_illustrate(self_, item, output_dir, graph, config=config)

        mock_export = MagicMock(return_value=tmp_path / "export" / "batch_0001")
        mock_rsync = AsyncMock(return_value=True)

        with (
            patch("core.task_queue.rate_limits.get_imagen_daily_tracker", return_value=tracker),
            patch("workflows.output.illustrate.illustrate_graph", _mock_illustrate_graph()),
            patch.object(type(workflow), "_illustrate_article", spy_illustrate),
            patch("core.task_queue.workflows.illustrate_and_export.export_batch", mock_export),
            patch("core.task_queue.workflows.illustrate_and_export.rsync_batch", mock_rsync),
        ):
            result = await workflow.run(
                task,
                checkpoint_callback=MagicMock(),
                resume_from=None,
                flush_checkpoints=AsyncMock(),
                update_items_callback=AsyncMock(),
            )

        assert result["status"] == "success"
        # First article: no config (no cached VI yet)
        assert configs_seen[0] is None
        # Second article: config with cached VI override
        assert configs_seen[1] is not None
        assert configs_seen[1].visual_identity_override is not None
        assert configs_seen[1].visual_identity_override.primary_style == "editorial watercolor"

    def test_properties(self, workflow):
        assert workflow.task_type == "illustrate_and_export"
        assert workflow.phases == ["processing", "complete"]
