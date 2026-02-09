"""Tests for illustrate workflow Pydantic schemas.

Covers field_validator edge cases for list fields that may receive
strings from LLM structured output (A1 fix).

Note: VisionReviewResult, ImageCompareResult, and HeaderAppositenessResult
were removed in the over-generation + selection refactor. The shared
_parse_json_string_list validator is still exercised via VisualIdentity
and CandidateBrief list fields in test_two_pass_schemas.py.
"""
