"""Unit tests for routing decision module."""

from core.scraping.pdf.analysis import DocumentAnalysis, DocumentComplexity
from core.scraping.pdf.routing import (
    RouteDecision,
    _confidence_for_cpu,
    determine_route,
    get_dynamic_batch_size,
)


class TestRouteDecision:
    """Tests for RouteDecision dataclass."""

    def test_route_decision_fields(self):
        """Verify RouteDecision has all expected fields."""
        decision = RouteDecision(
            queue="cpu",
            confidence=0.95,
            recommended_batch_size=12,
        )

        assert decision.queue == "cpu"
        assert decision.confidence == 0.95
        assert decision.recommended_batch_size == 12


class TestDetermineRoute:
    """Tests for determine_route function."""

    def test_scanned_document_routes_to_gpu(self):
        """Scanned documents always need GPU for OCR."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,  # Even if light complexity
            page_count=10,
            has_images=False,
            has_tables=False,
            is_scanned=True,  # Key: scanned
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=False,
        )

        route = determine_route(analysis)

        assert route.queue == "gpu"
        assert route.confidence == 0.95
        assert route.recommended_batch_size == 4

    def test_heavy_complexity_routes_to_gpu(self):
        """Heavy complexity documents route to GPU."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.HEAVY,
            page_count=50,
            has_images=True,
            has_tables=True,
            is_scanned=False,
            avg_image_ratio=0.4,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        route = determine_route(analysis)

        assert route.queue == "gpu"
        assert route.confidence == 0.85
        assert route.recommended_batch_size == 4

    def test_mixed_complexity_routes_to_gpu(self):
        """Mixed complexity documents route to GPU."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.MIXED,
            page_count=30,
            has_images=True,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.15,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        route = determine_route(analysis)

        assert route.queue == "gpu"
        assert route.confidence == 0.80
        assert route.recommended_batch_size == 8

    def test_light_complexity_high_confidence_routes_to_cpu(self):
        """Light complexity with high confidence routes to CPU."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=20,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,  # No multi-column = high confidence
            has_extractable_text=True,
        )

        route = determine_route(analysis)

        assert route.queue == "cpu"
        assert route.confidence >= 0.85
        assert route.recommended_batch_size == 12

    def test_light_complexity_low_confidence_routes_to_gpu(self):
        """Light complexity with low confidence falls back to GPU."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=10,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=5,  # Half pages are multi-column = low confidence
            has_extractable_text=True,
        )

        route = determine_route(analysis)

        # With 5/10 multi-column pages, confidence = 0.5 < 0.85
        assert route.queue == "gpu"
        assert route.confidence == 0.90  # GPU fallback confidence
        assert route.recommended_batch_size == 12


class TestConfidenceForCpu:
    """Tests for _confidence_for_cpu helper function."""

    def test_zero_confidence_for_scanned(self):
        """Scanned documents get zero CPU confidence."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=10,
            has_images=False,
            has_tables=False,
            is_scanned=True,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=False,
        )

        confidence = _confidence_for_cpu(analysis)
        assert confidence == 0.0

    def test_full_confidence_for_simple_text(self):
        """Simple text documents get full CPU confidence."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=20,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        confidence = _confidence_for_cpu(analysis)
        assert confidence == 1.0

    def test_reduced_confidence_for_multi_column(self):
        """Multi-column pages reduce CPU confidence."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=10,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=3,  # 3/10 = 30% multi-column
            has_extractable_text=True,
        )

        confidence = _confidence_for_cpu(analysis)
        assert confidence == 0.7  # 1.0 - 3/10

    def test_confidence_clamped_to_valid_range(self):
        """Confidence is clamped between 0 and 1."""
        # Edge case: more multi-column pages than total (shouldn't happen but test bounds)
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=5,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=10,  # More than page count (edge case)
            has_extractable_text=True,
        )

        confidence = _confidence_for_cpu(analysis)
        assert confidence >= 0.0
        assert confidence <= 1.0


class TestGetDynamicBatchSize:
    """Tests for get_dynamic_batch_size function."""

    def test_heavy_complexity_gets_smaller_batches(self):
        """Heavy complexity documents get smaller batch sizes."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.HEAVY,
            page_count=100,
            has_images=True,
            has_tables=True,
            is_scanned=False,
            avg_image_ratio=0.4,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        batch_size = get_dynamic_batch_size(analysis)

        # VRAM per page for HEAVY is 0.3GB, with 14GB budget: 14/0.3 ≈ 46
        # Capped at min(46, 100, 16) = 16
        assert batch_size == 16

    def test_light_complexity_gets_larger_batches(self):
        """Light complexity documents get larger batch sizes."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=100,
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        batch_size = get_dynamic_batch_size(analysis)

        # VRAM per page for LIGHT is 0.08GB, with 14GB budget: 14/0.08 = 175
        # Capped at min(175, 100, 16) = 16
        assert batch_size == 16

    def test_batch_size_respects_page_count(self):
        """Batch size respects document page count."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.LIGHT,
            page_count=5,  # Small document
            has_images=False,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.0,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        batch_size = get_dynamic_batch_size(analysis)

        # Should be capped at page count
        assert batch_size == 5

    def test_batch_size_with_custom_vram_budget(self):
        """Batch size adjusts to custom VRAM budget."""
        analysis = DocumentAnalysis(
            complexity=DocumentComplexity.MIXED,
            page_count=100,
            has_images=True,
            has_tables=False,
            is_scanned=False,
            avg_image_ratio=0.15,
            multi_column=False,
            multi_column_pages=0,
            has_extractable_text=True,
        )

        # With smaller VRAM budget (8GB)
        batch_size = get_dynamic_batch_size(analysis, vram_budget_gb=8.0)

        # VRAM per page for MIXED is 0.15GB, with 8GB budget: 8/0.15 ≈ 53
        # Capped at min(53, 100, 16) = 16
        assert batch_size == 16

        # With very small VRAM budget (1GB)
        batch_size = get_dynamic_batch_size(analysis, vram_budget_gb=1.0)

        # 1/0.15 ≈ 6
        # Capped at min(6, 100, 16) = 6
        assert batch_size == 6
