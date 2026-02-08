"""Tests for image selection with weighted rubric (A2)."""

from core.images.selection import ImageSelection


class TestImageSelectionSchema:
    """Test ImageSelection schema has per-criterion scores."""

    def test_all_score_fields_present(self):
        result = ImageSelection(
            selected_index=0,
            brief_compliance_score=4,
            mood_score=3,
            quality_score=5,
            relevance_score=2,
            reasoning="Best match",
        )
        assert result.brief_compliance_score == 4
        assert result.mood_score == 3
        assert result.quality_score == 5
        assert result.relevance_score == 2

    def test_score_validation_min(self):
        import pytest

        with pytest.raises(Exception):
            ImageSelection(
                selected_index=0,
                brief_compliance_score=0,  # Below minimum
                mood_score=3,
                quality_score=5,
                relevance_score=2,
                reasoning="Test",
            )

    def test_score_validation_max(self):
        import pytest

        with pytest.raises(Exception):
            ImageSelection(
                selected_index=0,
                brief_compliance_score=6,  # Above maximum
                mood_score=3,
                quality_score=5,
                relevance_score=2,
                reasoning="Test",
            )
