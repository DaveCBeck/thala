"""Tests for post-enhancement methodology transparency."""

from core.task_queue.workflows.lit_review_full.phases.methodology import (
    append_editorial_summary,
)


def _sample_report():
    """A minimal lit review markdown with a methodology section."""
    return """# Literature Review

## Introduction

This review examines...

## Methodology

Papers were identified using OpenAlex...

## Theme 1: Economics

Discussion of economics...

## Discussion

Overall findings...
"""


class TestAppendEditorialSummary:
    def test_appends_before_next_heading(self):
        report = _sample_report()
        enhance_result = {
            "editing_state": {
                "section_enhancements": [
                    {
                        "section_id": "s1",
                        "section_heading": "Theme 1",
                        "citations_added": ["@a", "@b", "@c"],
                        "citations_removed": [],
                        "confidence": 0.9,
                        "original_word_count": 500,
                        "enhanced_word_count": 520,
                        "success": True,
                    },
                ],
            },
        }
        result = append_editorial_summary(report, enhance_result)

        assert "### Editorial Process" in result
        # Editorial section should appear between Methodology and Theme 1
        meth_pos = result.index("## Methodology")
        editorial_pos = result.index("### Editorial Process")
        theme_pos = result.index("## Theme 1")
        assert meth_pos < editorial_pos < theme_pos

    def test_no_changes_returns_unchanged(self):
        report = _sample_report()
        enhance_result = {"editing_state": {"section_enhancements": []}}
        result = append_editorial_summary(report, enhance_result)
        assert result == report

    def test_cosmetic_changes_returns_unchanged(self):
        report = _sample_report()
        enhance_result = {
            "editing_state": {
                "section_enhancements": [
                    {
                        "section_id": "s1",
                        "section_heading": "Theme 1",
                        "citations_added": [],
                        "citations_removed": [],
                        "confidence": 0.95,
                        "original_word_count": 500,
                        "enhanced_word_count": 510,
                        "success": True,
                    },
                ],
            },
        }
        result = append_editorial_summary(report, enhance_result)
        assert "### Editorial Process" not in result

    def test_no_editing_state(self):
        report = _sample_report()
        result = append_editorial_summary(report, {})
        assert result == report

    def test_no_methodology_heading(self):
        """Falls back to appending at end if no methodology heading found."""
        report = "# Review\n\n## Introduction\n\nSome text.\n"
        enhance_result = {
            "editing_state": {
                "section_enhancements": [
                    {
                        "section_id": "s1",
                        "section_heading": "Intro",
                        "citations_added": ["@a", "@b"],
                        "citations_removed": [],
                        "confidence": 0.9,
                        "original_word_count": 500,
                        "enhanced_word_count": 520,
                        "success": True,
                    },
                ],
            },
        }
        result = append_editorial_summary(report, enhance_result)
        assert "### Editorial Process" in result
        assert result.endswith("\n")
