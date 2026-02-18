"""Tests for editorial transparency summarisation."""

from workflows.enhance.editing.transparency import (
    summarise_editorial_changes,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_CITATION_CHANGES,
    SIGNIFICANT_WORD_DELTA_PCT,
)


def _make_enhancement(
    *,
    citations_added=None,
    citations_removed=None,
    confidence=0.9,
    original_word_count=500,
    enhanced_word_count=510,
    success=True,
) -> dict:
    return {
        "section_id": "s1",
        "section_heading": "Theme 1",
        "citations_added": citations_added or [],
        "citations_removed": citations_removed or [],
        "confidence": confidence,
        "original_word_count": original_word_count,
        "enhanced_word_count": enhanced_word_count,
        "success": success,
    }


class TestSummariseEditorialChanges:
    def test_empty_enhancements(self):
        assert summarise_editorial_changes([]) == ""

    def test_no_significant_changes(self):
        """Cosmetic edits (no citations, high confidence, small delta) produce no summary."""
        enhancements = [
            _make_enhancement(
                confidence=0.95,
                original_word_count=500,
                enhanced_word_count=510,
            ),
        ]
        assert summarise_editorial_changes(enhancements) == ""

    def test_citation_additions(self):
        enhancements = [
            _make_enhancement(citations_added=["@key1", "@key2", "@key3"]),
            _make_enhancement(citations_added=["@key4"]),
        ]
        result = summarise_editorial_changes(enhancements)
        assert "4 citations" in result
        assert "2 sections" in result
        assert "editorial enhancement" in result

    def test_citation_removals(self):
        enhancements = [
            _make_enhancement(citations_removed=["@old1", "@old2"]),
        ]
        result = summarise_editorial_changes(enhancements)
        assert "removal of 2 citations" in result

    def test_low_confidence_sections(self):
        enhancements = [
            _make_enhancement(confidence=0.5),
            _make_enhancement(confidence=0.6),
        ]
        result = summarise_editorial_changes(enhancements)
        assert "2 sections were" in result
        assert "confidence was below threshold" in result

    def test_single_low_confidence(self):
        enhancements = [
            _make_enhancement(confidence=0.5),
        ]
        result = summarise_editorial_changes(enhancements)
        assert "1 section was" in result

    def test_large_word_count_change(self):
        enhancements = [
            _make_enhancement(original_word_count=500, enhanced_word_count=700),
        ]
        result = summarise_editorial_changes(enhancements)
        assert "word count changed" in result

    def test_failed_enhancements_excluded(self):
        enhancements = [
            _make_enhancement(citations_added=["@key1"], success=False),
        ]
        assert summarise_editorial_changes(enhancements) == ""

    def test_overall_word_delta_percentage(self):
        enhancements = [
            _make_enhancement(
                citations_added=["@key1"],
                original_word_count=1000,
                enhanced_word_count=1080,
            ),
        ]
        result = summarise_editorial_changes(enhancements)
        assert "8%" in result

    def test_constants_are_sensible(self):
        """Verify threshold constants are within expected ranges."""
        assert 0.0 < LOW_CONFIDENCE_THRESHOLD < 1.0
        assert MIN_CITATION_CHANGES >= 1
        assert 0.0 < SIGNIFICANT_WORD_DELTA_PCT < 1.0
