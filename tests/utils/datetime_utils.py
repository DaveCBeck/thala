"""
Datetime utilities for test scripts.

Provides consistent handling of datetime parsing, timezone normalization,
and duration formatting across all test files.
"""

from datetime import datetime


def parse_iso_datetime(dt: str | datetime) -> datetime:
    """Parse ISO datetime string, handling Z suffix and timezone.

    Args:
        dt: Either a datetime object or ISO format string

    Returns:
        Parsed datetime object (may have timezone info)
    """
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, str):
        # Handle Z suffix (UTC indicator)
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    raise TypeError(f"Expected str or datetime, got {type(dt)}")


def make_naive(dt: datetime) -> datetime:
    """Strip timezone info for comparison.

    Args:
        dt: Datetime object (with or without timezone)

    Returns:
        Naive datetime (no timezone info)
    """
    if hasattr(dt, "replace") and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def calculate_duration_seconds(
    start: str | datetime,
    end: str | datetime,
) -> float:
    """Calculate duration in seconds between two datetimes.

    Args:
        start: Start datetime (string or datetime)
        end: End datetime (string or datetime)

    Returns:
        Duration in seconds
    """
    start_dt = make_naive(parse_iso_datetime(start))
    end_dt = make_naive(parse_iso_datetime(end))
    return (end_dt - start_dt).total_seconds()


def format_duration(
    start: str | datetime,
    end: str | datetime,
) -> str:
    """Format duration between two datetimes as 'Xm Ys'.

    Args:
        start: Start datetime (string or datetime)
        end: End datetime (string or datetime)

    Returns:
        Formatted string like "5m 30s (330.0s total)"
    """
    try:
        duration = calculate_duration_seconds(start, end)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s ({duration:.1f}s total)"
    except Exception as e:
        return f"(error calculating: {e})"
