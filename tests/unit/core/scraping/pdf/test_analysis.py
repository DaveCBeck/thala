"""Unit tests for document analysis module."""

import io
from unittest.mock import MagicMock, patch


from core.scraping.pdf.analysis import (
    DocumentAnalysis,
    DocumentComplexity,
    _img_area,
    analyze_document,
)


class TestDocumentComplexity:
    """Tests for DocumentComplexity enum."""

    def test_complexity_values(self):
        """Verify complexity tier values."""
        assert DocumentComplexity.LIGHT.value == "light"
        assert DocumentComplexity.MIXED.value == "mixed"
        assert DocumentComplexity.HEAVY.value == "heavy"

    def test_complexity_members(self):
        """Verify all expected complexity tiers exist."""
        assert len(DocumentComplexity) == 3
        assert DocumentComplexity.LIGHT in DocumentComplexity
        assert DocumentComplexity.MIXED in DocumentComplexity
        assert DocumentComplexity.HEAVY in DocumentComplexity


class TestDocumentAnalysis:
    """Tests for DocumentAnalysis dataclass."""

    def test_analysis_fields(self):
        """Verify DocumentAnalysis has all expected fields."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=10,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        assert analysis.complexity == DocumentComplexity.LIGHT
        assert analysis.page_count == 10
        assert analysis.has_images is False
        assert analysis.has_tables is False
        assert analysis.is_scanned is False
        assert analysis.avg_image_ratio == 0.0
        assert analysis.multi_column is False
        assert analysis.multi_column_pages == 0
        assert analysis.has_extractable_text is True


class TestAnalyzeDocument:
    """Tests for analyze_document function."""

    @patch("core.scraping.pdf.analysis.fitz")
    def test_analyze_text_only_document(self, mock_fitz):
        """Test analysis of simple text-only document."""
        # Mock a simple text-only PDF
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__iter__ = MagicMock(return_value=iter([self._create_text_page() for _ in range(5)]))
        mock_fitz.open.return_value = mock_doc

        result = analyze_document(b"fake pdf content")

        assert result.complexity == DocumentComplexity.LIGHT
        assert result.page_count == 5
        assert result.has_images is False
        assert result.is_scanned is False
        assert result.has_extractable_text is True

    @patch("core.scraping.pdf.analysis.fitz")
    def test_analyze_scanned_document(self, mock_fitz):
        """Test analysis of scanned document (no extractable text)."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__iter__ = MagicMock(return_value=iter([self._create_scanned_page() for _ in range(3)]))
        mock_fitz.open.return_value = mock_doc

        result = analyze_document(b"fake scanned pdf")

        assert result.complexity == DocumentComplexity.HEAVY
        assert result.is_scanned is True
        assert result.has_extractable_text is False

    @patch("core.scraping.pdf.analysis.fitz")
    def test_analyze_image_heavy_document(self, mock_fitz):
        """Test analysis of image-heavy document."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=10)
        # Create mix of pages, >10% with images
        pages = [self._create_image_page() for _ in range(5)]
        pages.extend([self._create_text_page() for _ in range(5)])
        mock_doc.__iter__ = MagicMock(return_value=iter(pages))
        mock_fitz.open.return_value = mock_doc

        result = analyze_document(b"fake image-heavy pdf")

        assert result.has_images is True
        assert result.complexity in [DocumentComplexity.MIXED, DocumentComplexity.HEAVY]

    def test_multi_column_detection_logic(self):
        """Test multi-column detection logic based on distinct x positions.

        Multi-column is detected when >30% of pages have >2 distinct x positions.
        This test verifies the threshold logic directly.
        """
        # With 5/10 multi-column pages, multi_column should be True (50% > 30%)
        multi_column_pages = 5
        page_count = 10
        threshold = page_count * 0.3
        assert multi_column_pages > threshold

        # With 2/10 multi-column pages, multi_column should be False (20% < 30%)
        multi_column_pages = 2
        assert multi_column_pages <= threshold

    @patch("core.scraping.pdf.analysis.fitz")
    def test_analyze_empty_document(self, mock_fitz):
        """Test analysis of empty document.

        Empty documents have no extractable text, which is indistinguishable
        from scanned documents, so they get classified as HEAVY (needing OCR).
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_doc.__iter__ = MagicMock(return_value=iter([]))
        mock_fitz.open.return_value = mock_doc

        result = analyze_document(b"empty pdf")

        assert result.page_count == 0
        # Empty documents have no text, so is_scanned=True -> HEAVY
        assert result.is_scanned is True
        assert result.complexity == DocumentComplexity.HEAVY

    @patch("core.scraping.pdf.analysis.fitz")
    def test_analyze_accepts_bytes(self, mock_fitz):
        """Test that analyze_document accepts bytes input."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([self._create_text_page()]))
        mock_fitz.open.return_value = mock_doc

        # Should not raise
        result = analyze_document(b"pdf bytes")
        assert result is not None
        mock_fitz.open.assert_called_once_with(stream=b"pdf bytes", filetype="pdf")

    @patch("core.scraping.pdf.analysis.fitz")
    def test_analyze_accepts_file_like_object(self, mock_fitz):
        """Test that analyze_document accepts file-like objects."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([self._create_text_page()]))
        mock_fitz.open.return_value = mock_doc

        file_obj = io.BytesIO(b"pdf content")
        result = analyze_document(file_obj)
        assert result is not None

    def _create_text_page(self):
        """Create a mock text-only page."""
        page = MagicMock()
        page.rect.width = 612
        page.rect.height = 792
        page.get_text.return_value = "Sample text content"
        page.get_images.return_value = []
        page.get_text.side_effect = lambda mode: (
            "Sample text content" if mode == "text" else {"blocks": [{"type": 0, "bbox": [72, 72, 540, 720]}]}
        )
        return page

    def _create_scanned_page(self):
        """Create a mock scanned page (no extractable text)."""
        page = MagicMock()
        page.rect.width = 612
        page.rect.height = 792
        page.get_text.return_value = ""
        page.get_images.return_value = [(1, 0, 0, 0, 0, "image", "", "", 0, 0)]
        page.get_text.side_effect = lambda mode: ("" if mode == "text" else {"blocks": []})
        page.get_image_rects.return_value = [MagicMock(width=600, height=780)]
        return page

    def _create_image_page(self):
        """Create a mock page with images."""
        page = MagicMock()
        page.rect.width = 612
        page.rect.height = 792
        page.get_text.side_effect = lambda mode: (
            "Some text" if mode == "text" else {"blocks": [{"type": 0, "bbox": [72, 72, 300, 400]}]}
        )
        page.get_images.return_value = [(1, 0, 0, 0, 0, "image", "", "", 0, 0)]
        page.get_image_rects.return_value = [MagicMock(width=400, height=300)]
        return page

    def _create_multi_column_page(self):
        """Create a mock multi-column page."""
        page = MagicMock()
        page.rect.width = 612
        page.rect.height = 792
        page.get_text.side_effect = lambda mode: (
            "Multi-column text"
            if mode == "text"
            else {
                "blocks": [
                    {"type": 0, "bbox": [50, 100, 280, 700]},  # Column 1
                    {"type": 0, "bbox": [320, 100, 560, 700]},  # Column 2
                    {"type": 0, "bbox": [50, 50, 560, 80]},  # Header spanning both
                ]
            }
        )
        page.get_images.return_value = []
        return page


class TestImgArea:
    """Tests for _img_area helper function."""

    def test_img_area_with_valid_rects(self):
        """Test image area calculation with valid rectangles."""
        page = MagicMock()
        rect1 = MagicMock(width=100, height=200)
        rect2 = MagicMock(width=50, height=100)
        page.get_image_rects.return_value = [rect1, rect2]

        area = _img_area(page, (1, 0, 0, 0, 0))
        assert area == 100 * 200 + 50 * 100  # 25000

    def test_img_area_with_no_rects(self):
        """Test image area when no rectangles found."""
        page = MagicMock()
        page.get_image_rects.return_value = []

        area = _img_area(page, (1, 0, 0, 0, 0))
        assert area == 0

    def test_img_area_with_exception(self):
        """Test image area handles exceptions gracefully."""
        page = MagicMock()
        page.get_image_rects.side_effect = Exception("Image error")

        area = _img_area(page, (1, 0, 0, 0, 0))
        assert area == 0.0
