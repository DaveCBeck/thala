"""Validation utilities for MCP tools."""

from uuid import UUID


def parse_uuid_arg(arguments: dict, key: str = "id") -> UUID:
    """Parse UUID from arguments dict."""
    return UUID(arguments[key])


def parse_uuid_list_arg(arguments: dict, key: str = "source_ids") -> list[UUID]:
    """Parse list of UUIDs from arguments dict."""
    return [UUID(sid) for sid in arguments.get(key, [])]


def uuid_list_to_strings(uuids: list[UUID]) -> list[str]:
    """Convert UUID list to string list."""
    return [str(u) for u in uuids]
