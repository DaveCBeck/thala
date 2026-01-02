"""Validation utilities for LangChain tools."""


def clamp_limit(limit: int, min_val: int = 1, max_val: int = 50) -> int:
    """Clamp limit parameter to valid range."""
    return min(max(min_val, limit), max_val)
