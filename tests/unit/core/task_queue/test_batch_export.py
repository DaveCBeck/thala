"""Tests for batch export module."""

import json
from unittest.mock import patch

import pytest

from core.task_queue.workflows.shared.batch_export import export_batch, rewrite_image_paths


class TestRewriteImagePaths:
    def test_rewrites_absolute_to_relative(self):
        content = "![Header](/home/user/.thala/output/unillustrated_topic_abc_123/overview_images/header.png)"
        abs_dir = "/home/user/.thala/output/unillustrated_topic_abc_123"
        result = rewrite_image_paths(content, abs_dir)
        assert result == "![Header](./overview_images/header.png)"

    def test_rewrites_multiple_images(self):
        abs_dir = "/output/dir"
        content = (
            "![A](/output/dir/img1/a.png)\n"
            "text\n"
            "![B](/output/dir/img2/b.png)"
        )
        result = rewrite_image_paths(content, abs_dir)
        assert "![A](./img1/a.png)" in result
        assert "![B](./img2/b.png)" in result

    def test_leaves_unrelated_paths_alone(self):
        content = "![Other](/somewhere/else/image.png)"
        result = rewrite_image_paths(content, "/output/dir")
        assert result == content

    def test_handles_trailing_slash_in_dir(self):
        content = "![A](/output/dir/images/x.png)"
        result = rewrite_image_paths(content, "/output/dir/")
        assert result == "![A](./images/x.png)"


class TestExportBatch:
    @pytest.fixture
    def source_dir(self, tmp_path):
        """Create a source directory mimicking unillustrated_* output."""
        src = tmp_path / "source"
        src.mkdir()

        # Lit review
        (src / "lit_review.md").write_text("# Literature Review\n\nContent.")

        # Illustrated article with image
        abs_prefix = str(src)
        (src / "overview_illustrated.md").write_text(
            f"# Overview\n\n![Header]({abs_prefix}/overview_images/header.png)"
        )
        img_dir = src / "overview_images"
        img_dir.mkdir()
        (img_dir / "header.png").write_bytes(b"fake png")

        return src

    @pytest.fixture
    def manifest(self, source_dir):
        return {
            "topic": "Test Topic",
            "category": "science",
            "quality": "standard",
            "source_task_id": "task-123",
            "output_dir": str(source_dir),
            "articles": [
                {"id": "overview", "title": "Overview", "filename": "overview.md"},
            ],
        }

    @pytest.fixture
    def items(self, source_dir):
        return [
            {
                "id": "overview",
                "title": "Overview",
                "source_path": str(source_dir / "overview.md"),
                "illustrated": True,
                "illustrated_path": str(source_dir / "overview_illustrated.md"),
                "exported": False,
            },
        ]

    def test_creates_batch_with_manifest(self, source_dir, manifest, items, tmp_path):
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        with (
            patch("core.task_queue.workflows.shared.batch_export.EXPORT_DIR", export_dir),
            patch("core.task_queue.workflows.shared.batch_export.load_publication_config", return_value={"subdomain": "testpub", "publication_url": "testpub.substack.com"}),
            patch("core.task_queue.workflows.shared.batch_export.next_id", return_value=1),
        ):
            batch_dir = export_batch(source_dir, manifest, items, "science")

        assert batch_dir.exists()
        batch_manifest = json.loads((batch_dir / "manifest.json").read_text())
        assert batch_manifest["batch_id"] == "batch_0001"
        assert batch_manifest["publication_slug"] == "testpub"
        assert len(batch_manifest["articles"]) == 2  # overview + lit_review

    def test_rewrites_image_paths_in_exported_markdown(self, source_dir, manifest, items, tmp_path):
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        with (
            patch("core.task_queue.workflows.shared.batch_export.EXPORT_DIR", export_dir),
            patch("core.task_queue.workflows.shared.batch_export.load_publication_config", return_value={"subdomain": "testpub", "publication_url": "testpub.substack.com"}),
            patch("core.task_queue.workflows.shared.batch_export.next_id", return_value=1),
        ):
            batch_dir = export_batch(source_dir, manifest, items, "science")

        exported_md = (batch_dir / "overview_illustrated.md").read_text()
        assert str(source_dir) not in exported_md
        assert "./overview_images/header.png" in exported_md

    def test_copies_images_directory(self, source_dir, manifest, items, tmp_path):
        export_dir = tmp_path / "export"
        export_dir.mkdir()

        with (
            patch("core.task_queue.workflows.shared.batch_export.EXPORT_DIR", export_dir),
            patch("core.task_queue.workflows.shared.batch_export.load_publication_config", return_value={"subdomain": "testpub", "publication_url": "testpub.substack.com"}),
            patch("core.task_queue.workflows.shared.batch_export.next_id", return_value=1),
        ):
            batch_dir = export_batch(source_dir, manifest, items, "science")

        assert (batch_dir / "overview_images" / "header.png").exists()
