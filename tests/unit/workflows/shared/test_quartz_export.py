"""Tests for Quartz literature review export."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestSlugifyTopic:
    """Test _slugify_topic helper."""

    def test_basic_topic(self):
        from core.task_queue.workflows.shared.quartz_export import _slugify_topic

        assert _slugify_topic("Forest Decline Dynamics") == "forest-decline-dynamics"

    def test_topic_with_colon(self):
        from core.task_queue.workflows.shared.quartz_export import _slugify_topic

        result = _slugify_topic(
            "Forest Decline Dynamics: Drought, Fire, and Ecosystem Collapse"
        )
        assert result == "forest-decline-dynamics-drought-fire-and-ecosystem-collapse"

    def test_topic_with_special_characters(self):
        from core.task_queue.workflows.shared.quartz_export import _slugify_topic

        result = _slugify_topic("Rapamycin and mTOR: The Biology of Growth-Longevity Trade-offs")
        assert result == "rapamycin-and-mtor-the-biology-of-growth-longevity-trade-offs"

    def test_max_length_truncation(self):
        from core.task_queue.workflows.shared.quartz_export import _slugify_topic

        result = _slugify_topic("a " * 100, max_length=10)
        assert len(result) <= 10
        assert not result.endswith("-")

    def test_collapses_multiple_hyphens(self):
        from core.task_queue.workflows.shared.quartz_export import _slugify_topic

        result = _slugify_topic("Brain---Computer   Interfaces")
        assert "--" not in result


class TestExtractAbstract:
    """Test _extract_abstract helper."""

    def test_italic_abstract(self):
        from core.task_queue.workflows.shared.quartz_export import _extract_abstract

        content = (
            "# Literature Review: Test Topic\n\n"
            "*This is the abstract spanning\n"
            "multiple lines with important findings.*\n\n"
            "## 1. Introduction\n"
        )
        result = _extract_abstract(content)
        assert result.startswith("This is the abstract")
        assert result.endswith("important findings.")
        assert "\n" not in result  # collapsed to single line

    def test_no_abstract(self):
        from core.task_queue.workflows.shared.quartz_export import _extract_abstract

        content = "# Literature Review: Test\n\n## 1. Introduction\nSome text.\n"
        assert _extract_abstract(content) == ""

    def test_single_line_abstract(self):
        from core.task_queue.workflows.shared.quartz_export import _extract_abstract

        content = "# Literature Review: Test\n\n*Short abstract.*\n\n## 1. Introduction\n"
        assert _extract_abstract(content) == "Short abstract."


class TestBuildFrontmatter:
    """Test _build_frontmatter helper."""

    def test_basic_frontmatter(self):
        from core.task_queue.workflows.shared.quartz_export import _build_frontmatter

        result = _build_frontmatter(
            topic="Forest Decline",
            description="Trees are dying.",
            date="2026-02-22T04:39:27.650524",
            publication_slug="gaias-web",
            quality="quick",
        )
        assert result.startswith("---")
        assert result.endswith("---")
        assert 'title: "Forest Decline"' in result
        assert 'description: "Trees are dying."' in result
        assert "date: 2026-02-22" in result
        assert "- literature-review" in result
        assert "- gaias-web" in result
        assert "quality: quick" in result
        assert "draft: false" in result

    def test_escapes_quotes_in_title(self):
        from core.task_queue.workflows.shared.quartz_export import _build_frontmatter

        result = _build_frontmatter(
            topic='Topic with "quotes"',
            description="Desc",
            date="2026-01-01",
            publication_slug="native-state",
            quality="standard",
        )
        assert r'\"quotes\"' in result

    def test_date_truncation(self):
        from core.task_queue.workflows.shared.quartz_export import _build_frontmatter

        result = _build_frontmatter(
            topic="T",
            description="D",
            date="2026-02-22T12:00:00+00:00",
            publication_slug="gaias-web",
            quality="quick",
        )
        assert "date: 2026-02-22" in result
        assert "T12" not in result


class TestExportLitReviewToQuartz:
    """Test the main export_lit_review_to_quartz function."""

    @pytest.mark.asyncio
    async def test_writes_file_to_correct_path(self, tmp_path):
        from core.task_queue.workflows.shared.quartz_export import (
            export_lit_review_to_quartz,
        )

        content = (
            "# Literature Review: Forest Decline\n\n"
            "*Trees are dying from drought.*\n\n"
            "## 1. Introduction\nBody text.\n"
        )

        with patch(
            "core.task_queue.workflows.shared.quartz_export.QUARTZ_CONTENT_DIR",
            tmp_path,
        ):
            result = await export_lit_review_to_quartz(
                content=content,
                topic="Forest Decline",
                category="gaias web",
                generated_at="2026-02-22T04:39:27",
                quality="quick",
            )

        assert result is not None
        assert result.parent.name == "gaias-web"
        assert result.name == "forest-decline.md"
        assert result.exists()

        text = result.read_text()
        assert text.startswith("---")
        assert 'title: "Forest Decline"' in text
        assert "Trees are dying from drought." in text
        assert "## 1. Introduction" in text

    @pytest.mark.asyncio
    async def test_creates_publication_directory(self, tmp_path):
        from core.task_queue.workflows.shared.quartz_export import (
            export_lit_review_to_quartz,
        )

        content = "# Literature Review: Topic\n\n*Abstract.*\n\n## Body\n"

        with patch(
            "core.task_queue.workflows.shared.quartz_export.QUARTZ_CONTENT_DIR",
            tmp_path,
        ):
            result = await export_lit_review_to_quartz(
                content=content,
                topic="Topic",
                category="native state",
                generated_at="2026-01-01",
                quality="standard",
            )

        assert result is not None
        assert (tmp_path / "native-state").is_dir()

    @pytest.mark.asyncio
    async def test_unknown_category_returns_none(self, tmp_path):
        from core.task_queue.workflows.shared.quartz_export import (
            export_lit_review_to_quartz,
        )

        with patch(
            "core.task_queue.workflows.shared.quartz_export.QUARTZ_CONTENT_DIR",
            tmp_path,
        ):
            result = await export_lit_review_to_quartz(
                content="# Review\n\n*Abstract.*\n",
                topic="Topic",
                category="unknown category",
                generated_at="2026-01-01",
                quality="quick",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_case_insensitive_category(self, tmp_path):
        from core.task_queue.workflows.shared.quartz_export import (
            export_lit_review_to_quartz,
        )

        content = "# Literature Review: Topic\n\n*Abstract.*\n\n## Body\n"

        with patch(
            "core.task_queue.workflows.shared.quartz_export.QUARTZ_CONTENT_DIR",
            tmp_path,
        ):
            result = await export_lit_review_to_quartz(
                content=content,
                topic="Topic",
                category="Gaias Web",
                generated_at="2026-01-01",
                quality="quick",
            )

        assert result is not None
        assert result.parent.name == "gaias-web"

    @pytest.mark.asyncio
    async def test_idempotent_overwrite(self, tmp_path):
        from core.task_queue.workflows.shared.quartz_export import (
            export_lit_review_to_quartz,
        )

        content = "# Literature Review: Topic\n\n*Abstract v1.*\n\n## Body\n"

        with patch(
            "core.task_queue.workflows.shared.quartz_export.QUARTZ_CONTENT_DIR",
            tmp_path,
        ):
            path1 = await export_lit_review_to_quartz(
                content=content,
                topic="Topic",
                category="gaias web",
                generated_at="2026-01-01",
                quality="quick",
            )

            content2 = "# Literature Review: Topic\n\n*Abstract v2.*\n\n## Body\n"
            path2 = await export_lit_review_to_quartz(
                content=content2,
                topic="Topic",
                category="gaias web",
                generated_at="2026-01-02",
                quality="standard",
            )

        assert path1 == path2
        text = path2.read_text()
        assert "Abstract v2." in text
        assert "Abstract v1." not in text
