"""Tests for illustrate workflow Pydantic schemas.

Covers field_validator edge cases for list fields that may receive
strings from LLM structured output (A1 fix).
"""

import json

from workflows.output.illustrate.schemas import (
    ImageCompareResult,
    VisionReviewResult,
)


class TestVisionReviewResultIssuesValidator:
    """Test VisionReviewResult.issues field_validator handles all input types."""

    def _make(self, issues_value):
        return VisionReviewResult(
            fits_context=True,
            has_substantive_errors=False,
            has_minor_issues=False,
            issues=issues_value,
            recommendation="accept",
        )

    def test_list_passthrough(self):
        result = self._make(["issue one", "issue two"])
        assert result.issues == ["issue one", "issue two"]

    def test_empty_list(self):
        result = self._make([])
        assert result.issues == []

    def test_none_becomes_empty_list(self):
        result = self._make(None)
        assert result.issues == []

    def test_json_string_array(self):
        json_str = json.dumps(["text overlap", "low contrast"])
        result = self._make(json_str)
        assert result.issues == ["text overlap", "low contrast"]

    def test_plain_string_becomes_single_item_list(self):
        result = self._make("some issue description")
        assert result.issues == ["some issue description"]

    def test_empty_string_becomes_empty_list(self):
        result = self._make("")
        assert result.issues == []

    def test_whitespace_string_becomes_empty_list(self):
        result = self._make("   ")
        assert result.issues == []

    def test_json_string_non_list_becomes_single_item(self):
        """JSON string that parses but isn't a list stays as single string."""
        result = self._make('"just a string"')
        assert result.issues == ['"just a string"']


class TestImageCompareResultIssuesValidator:
    """Test ImageCompareResult.issues_with_selected field_validator."""

    def _make(self, issues_value):
        return ImageCompareResult(
            selected_candidate=1,
            reasoning="Best quality overall",
            issues_with_selected=issues_value,
        )

    def test_list_passthrough(self):
        result = self._make(["minor blur", "slight crop"])
        assert result.issues_with_selected == ["minor blur", "slight crop"]

    def test_empty_list(self):
        result = self._make([])
        assert result.issues_with_selected == []

    def test_none_becomes_empty_list(self):
        result = self._make(None)
        assert result.issues_with_selected == []

    def test_json_string_array(self):
        json_str = json.dumps(["blur on edges", "color balance off"])
        result = self._make(json_str)
        assert result.issues_with_selected == ["blur on edges", "color balance off"]

    def test_plain_string_becomes_single_item_list(self):
        result = self._make("slight blur in corner")
        assert result.issues_with_selected == ["slight blur in corner"]

    def test_empty_string_becomes_empty_list(self):
        result = self._make("")
        assert result.issues_with_selected == []

    def test_default_factory(self):
        """Test default_factory still works when field is omitted."""
        result = ImageCompareResult(
            selected_candidate=1,
            reasoning="test",
        )
        assert result.issues_with_selected == []
