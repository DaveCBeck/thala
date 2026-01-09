"""
Result display utilities for test scripts.

Provides consistent formatting for displaying workflow results,
including section headers, previews, timing, and error display.
"""

from datetime import datetime

from .datetime_utils import format_duration


def print_section_header(
    title: str,
    char: str = "=",
    width: int = 80,
) -> None:
    """Print a formatted section header.

    Args:
        title: Header title
        char: Character to use for the line (default: =)
        width: Total width of the line (default: 80)
    """
    print("\n" + char * width)
    print(title)
    print(char * width)


def print_subsection_header(title: str) -> None:
    """Print a subsection header.

    Args:
        title: Subsection title
    """
    print(f"\n--- {title} ---")


def safe_preview(
    text: str | None,
    max_chars: int = 1000,
    suffix: str = "\n\n... [truncated] ...",
) -> str:
    """Truncate text with suffix if needed.

    Args:
        text: Text to preview (can be None)
        max_chars: Maximum characters before truncation
        suffix: Suffix to add when truncated

    Returns:
        Truncated text or empty string if None
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def print_timing(
    started: str | datetime | None,
    completed: str | datetime | None,
) -> None:
    """Print formatted timing information.

    Args:
        started: Start time (string or datetime)
        completed: End time (string or datetime)
    """
    if started and completed:
        duration_str = format_duration(started, completed)
        print(f"Duration: {duration_str}")


def print_key_value(key: str, value, indent: int = 0) -> None:
    """Print a key-value pair with optional indentation.

    Args:
        key: Label for the value
        value: Value to display
        indent: Number of spaces to indent (default: 0)
    """
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def print_list_preview(
    items: list,
    title: str,
    max_items: int = 5,
    format_item=None,
    indent: int = 2,
) -> None:
    """Print a preview of a list with optional formatting.

    Args:
        items: List to preview
        title: Title for the list (with count)
        max_items: Maximum items to show
        format_item: Optional function to format each item (default: str)
        indent: Number of spaces to indent items (default: 2)
    """
    if not items:
        return

    print(f"\n{title} ({len(items)}):")
    prefix = " " * indent

    if format_item is None:
        format_item = str

    for item in items[:max_items]:
        print(f"{prefix}- {format_item(item)}")

    if len(items) > max_items:
        print(f"{prefix}... and {len(items) - max_items} more")


def print_errors(errors: list[dict]) -> None:
    """Print workflow errors in consistent format.

    Args:
        errors: List of error dicts with 'phase'/'node' and 'error' keys
    """
    if not errors:
        return

    print(f"\n--- Errors ({len(errors)}) ---")
    for err in errors:
        # Support both 'phase' and 'node' keys
        location = err.get("phase") or err.get("node", "unknown")
        error_msg = err.get("error", "unknown")
        print(f"  [{location}]: {error_msg}")


def print_storage_info(
    store_id: str | None = None,
    zotero_key: str | None = None,
    zotero_keys: dict | None = None,
    es_ids: dict | None = None,
    langsmith_run_id: str | None = None,
) -> None:
    """Print storage and tracing information.

    Args:
        store_id: Store record ID
        zotero_key: Single Zotero key
        zotero_keys: Dict of Zotero keys
        es_ids: Elasticsearch IDs
        langsmith_run_id: LangSmith run ID
    """
    if not any([store_id, zotero_key, zotero_keys, es_ids, langsmith_run_id]):
        return

    print(f"\n--- Storage & Tracing ---")
    if store_id:
        print(f"Store Record ID: {store_id}")
    if zotero_key:
        print(f"Zotero Key: {zotero_key}")
    if zotero_keys:
        print(f"Zotero items created: {len(zotero_keys)}")
    if es_ids:
        print(f"Elasticsearch records: {len(es_ids)}")
    if langsmith_run_id:
        print(f"LangSmith Run ID: {langsmith_run_id}")


def print_report_preview(
    report: str | None,
    title: str = "Final Report",
    max_chars: int = 1000,
) -> None:
    """Print a report preview with word count.

    Args:
        report: Report content
        title: Section title
        max_chars: Maximum characters to show
    """
    if not report:
        return

    print(f"\n--- {title} ---")
    word_count = len(report.split())
    print(f"Length: {len(report)} chars ({word_count} words)")
    print(safe_preview(report, max_chars))
