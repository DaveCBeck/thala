"""Response formatting utilities for MCP tools."""


def format_single_record(record) -> dict:
    """Format single record to JSON dict."""
    return record.model_dump(mode="json")


def format_search_results(records: list, count_key: str = "count") -> dict:
    """Format list of records with count."""
    return {
        count_key: len(records),
        "results": [r.model_dump(mode="json") for r in records],
    }
