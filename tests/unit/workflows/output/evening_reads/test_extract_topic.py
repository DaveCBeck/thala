"""Tests for _extract_topic() helper in evening_reads workflow."""

import pytest

from workflows.output.evening_reads import _extract_topic


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("# AI in Healthcare", "AI in Healthcare"),
        ("## Sub heading", "Sub heading"),
        ("### Triple heading", "Triple heading"),
        ("No heading here", "Untitled"),
        ("", "Untitled"),
        ("  # Indented heading", "Indented heading"),
        ("Some preamble\n# Actual Title\nMore text", "Actual Title"),
        ("#", ""),
    ],
    ids=[
        "h1_heading",
        "h2_heading",
        "h3_heading",
        "no_heading",
        "empty_string",
        "indented_heading",
        "heading_after_preamble",
        "hash_only_no_text",
    ],
)
def test_extract_topic(input_text: str, expected: str):
    assert _extract_topic(input_text) == expected
