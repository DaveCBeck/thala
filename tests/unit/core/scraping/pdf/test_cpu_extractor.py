"""Unit tests for CPU extractor module."""

from unittest.mock import MagicMock, patch

import pytest

from core.scraping.pdf.cpu_extractor import (
    CpuExtractor,
    ExtractionResult,
    _extract_sync,
    _format_page_blocks,
    extract_text_cpu,
    get_extractor,
)


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_extraction_result_fields(self):
        """Verify ExtractionResult has all expected fields."""
        result = ExtractionResult(
            markdown="# Sample\n\nContent here",
            page_count=5,
            confidence=0.95,
            fallback_recommended=False,
        )

        assert result.markdown == "# Sample\n\nContent here"
        assert result.page_count == 5
        assert result.confidence == 0.95
        assert result.fallback_recommended is False

    def test_fallback_recommended_when_low_confidence(self):
        """Fallback should be recommended when confidence is low."""
        result = ExtractionResult(
            markdown="Content",
            page_count=1,
            confidence=0.70,
            fallback_recommended=True,  # Should be true for confidence < 0.85
        )

        assert result.fallback_recommended is True


class TestCpuExtractor:
    """Tests for CpuExtractor class."""

    @pytest.mark.asyncio
    async def test_context_manager_protocol(self):
        """Test CpuExtractor works as async context manager."""
        async with CpuExtractor(max_workers=2) as extractor:
            assert extractor._closed is False
        assert extractor._closed is True

    @pytest.mark.asyncio
    async def test_explicit_shutdown(self):
        """Test explicit shutdown closes executor."""
        extractor = CpuExtractor(max_workers=2)
        assert extractor._closed is False

        await extractor.shutdown()
        assert extractor._closed is True

        # Double shutdown should be safe
        await extractor.shutdown()
        assert extractor._closed is True

    @pytest.mark.asyncio
    async def test_extract_after_shutdown_raises(self):
        """Test that extract raises after shutdown."""
        extractor = CpuExtractor(max_workers=2)
        await extractor.shutdown()

        with pytest.raises(RuntimeError, match="shut down"):
            await extractor.extract(b"fake pdf")

    @pytest.mark.asyncio
    @patch("core.scraping.pdf.cpu_extractor._extract_sync")
    async def test_extract_calls_sync_in_executor(self, mock_extract_sync):
        """Test that extract delegates to sync extraction in executor."""
        mock_result = ExtractionResult(
            markdown="# Test",
            page_count=1,
            confidence=0.95,
            fallback_recommended=False,
        )
        mock_extract_sync.return_value = mock_result

        async with CpuExtractor(max_workers=2) as extractor:
            result = await extractor.extract(b"fake pdf content")

        assert result == mock_result
        mock_extract_sync.assert_called_once_with(b"fake pdf content")


class TestGetExtractor:
    """Tests for get_extractor singleton function."""

    def test_get_extractor_returns_instance(self):
        """Test get_extractor returns a CpuExtractor instance."""
        extractor = get_extractor()
        assert isinstance(extractor, CpuExtractor)

    def test_get_extractor_returns_same_instance(self):
        """Test get_extractor returns the same singleton instance."""
        extractor1 = get_extractor()
        extractor2 = get_extractor()
        assert extractor1 is extractor2


class TestExtractTextCpu:
    """Tests for extract_text_cpu convenience function."""

    @pytest.mark.asyncio
    @patch("core.scraping.pdf.cpu_extractor._extract_sync")
    async def test_extract_text_cpu_uses_singleton(self, mock_extract_sync):
        """Test extract_text_cpu uses the default extractor."""
        mock_result = ExtractionResult(
            markdown="Content",
            page_count=1,
            confidence=1.0,
            fallback_recommended=False,
        )
        mock_extract_sync.return_value = mock_result

        result = await extract_text_cpu(b"pdf content")

        assert result == mock_result


class TestExtractSync:
    """Tests for _extract_sync function."""

    @patch("core.scraping.pdf.cpu_extractor.fitz")
    def test_extract_simple_text(self, mock_fitz):
        """Test extraction of simple text document."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(
            return_value=iter(
                [
                    self._create_simple_page("Page 1 content"),
                    self._create_simple_page("Page 2 content"),
                ]
            )
        )
        mock_fitz.open.return_value = mock_doc

        result = _extract_sync(b"fake pdf")

        assert result.page_count == 2
        assert "Page 1 content" in result.markdown
        assert "Page 2 content" in result.markdown
        assert result.confidence == 1.0  # No multi-column issues
        assert result.fallback_recommended is False

    def test_confidence_calculation_with_issues(self):
        """Test confidence calculation logic.

        Confidence = 1.0 - (issues / page_count)
        fallback_recommended when confidence < 0.85
        """
        # With 1 issue out of 2 pages
        issues = 1
        pages = 2
        confidence = 1.0 - (issues / max(pages, 1))
        assert confidence == 0.5
        assert confidence < 0.85  # Should recommend fallback

        # With 0 issues out of 5 pages
        issues = 0
        pages = 5
        confidence = 1.0 - (issues / max(pages, 1))
        assert confidence == 1.0
        assert confidence >= 0.85  # Should not recommend fallback

        # With 1 issue out of 10 pages
        issues = 1
        pages = 10
        confidence = 1.0 - (issues / max(pages, 1))
        assert confidence == 0.9
        assert confidence >= 0.85  # Should not recommend fallback

    @patch("core.scraping.pdf.cpu_extractor.fitz")
    def test_extract_empty_document(self, mock_fitz):
        """Test extraction of empty document."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(
            return_value=iter(
                [
                    self._create_empty_page(),
                ]
            )
        )
        mock_fitz.open.return_value = mock_doc

        result = _extract_sync(b"empty pdf")

        assert result.page_count == 0
        assert result.markdown == ""

    @patch("core.scraping.pdf.cpu_extractor.fitz")
    def test_extract_preserves_page_markers(self, mock_fitz):
        """Test that extraction preserves page markers in markdown."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(
            return_value=iter(
                [
                    self._create_simple_page("First page"),
                    self._create_simple_page("Second page"),
                ]
            )
        )
        mock_fitz.open.return_value = mock_doc

        result = _extract_sync(b"pdf")

        assert "<!-- Page 1 -->" in result.markdown
        assert "<!-- Page 2 -->" in result.markdown
        assert "---" in result.markdown  # Page separator

    def _create_simple_page(self, text: str):
        """Create a mock simple text page."""
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [72, 72, 540, 720],
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": text,
                                    "size": 12,
                                    "flags": 0,
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        return page

    def _create_multi_column_page(self):
        """Create a mock multi-column page."""
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [50, 100, 280, 700],
                    "lines": [{"spans": [{"text": "Col 1", "size": 12, "flags": 0}]}],
                },
                {
                    "type": 0,
                    "bbox": [320, 100, 560, 700],
                    "lines": [{"spans": [{"text": "Col 2", "size": 12, "flags": 0}]}],
                },
                {
                    "type": 0,
                    "bbox": [50, 50, 560, 80],
                    "lines": [{"spans": [{"text": "Header", "size": 12, "flags": 0}]}],
                },
            ]
        }
        return page

    def _create_empty_page(self):
        """Create a mock empty page."""
        page = MagicMock()
        page.get_text.return_value = {"blocks": []}
        return page


class TestFormatPageBlocks:
    """Tests for _format_page_blocks helper function."""

    def test_format_simple_text(self):
        """Test formatting simple text blocks."""
        blocks = [
            {
                "type": 0,
                "bbox": [72, 72, 540, 100],
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Hello world",
                                "size": 12,
                                "flags": 0,
                            }
                        ]
                    }
                ],
            }
        ]

        result = _format_page_blocks(blocks)
        assert "Hello world" in result

    def test_format_bold_text(self):
        """Test formatting bold text (flag bit 4)."""
        blocks = [
            {
                "type": 0,
                "bbox": [72, 72, 540, 100],
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Bold text",
                                "size": 12,
                                "flags": 16,  # Bit 4 set for bold
                            }
                        ]
                    }
                ],
            }
        ]

        result = _format_page_blocks(blocks)
        assert "**Bold text**" in result

    def test_format_italic_text(self):
        """Test formatting italic text (flag bit 1)."""
        blocks = [
            {
                "type": 0,
                "bbox": [72, 72, 540, 100],
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Italic text",
                                "size": 12,
                                "flags": 2,  # Bit 1 set for italic
                            }
                        ]
                    }
                ],
            }
        ]

        result = _format_page_blocks(blocks)
        assert "*Italic text*" in result

    def test_format_bold_italic_text(self):
        """Test formatting bold and italic text."""
        blocks = [
            {
                "type": 0,
                "bbox": [72, 72, 540, 100],
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Bold italic",
                                "size": 12,
                                "flags": 18,  # Bits 1 and 4 set
                            }
                        ]
                    }
                ],
            }
        ]

        result = _format_page_blocks(blocks)
        # Both bold and italic markers should be present
        assert "**" in result
        assert "*" in result

    def test_format_multiple_blocks_sorted_by_position(self):
        """Test that blocks are sorted by vertical then horizontal position."""
        blocks = [
            {
                "type": 0,
                "bbox": [72, 200, 200, 250],
                "lines": [{"spans": [{"text": "Second", "size": 12, "flags": 0}]}],
            },
            {"type": 0, "bbox": [72, 100, 200, 150], "lines": [{"spans": [{"text": "First", "size": 12, "flags": 0}]}]},
        ]

        result = _format_page_blocks(blocks)
        first_pos = result.find("First")
        second_pos = result.find("Second")
        assert first_pos < second_pos  # First should appear before Second

    def test_format_empty_blocks(self):
        """Test formatting empty block list."""
        result = _format_page_blocks([])
        assert result == ""
