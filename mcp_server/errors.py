"""Error handling utilities for MCP tools."""

from typing import Any


class ToolError(Exception):
    """Base class for tool execution errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class NotFoundError(ToolError):
    """Record not found in store."""

    def __init__(self, store: str, record_id: str):
        super().__init__(
            f"Record not found. The UUID '{record_id}' does not exist in {store} store. "
            f"Use {store}.search to find records, or verify the UUID is correct.",
            {"store": store, "record_id": record_id},
        )


class ValidationError(ToolError):
    """Input validation failed."""

    def __init__(self, field: str, message: str):
        super().__init__(
            f"Validation error for '{field}': {message}",
            {"field": field},
        )


class StoreConnectionError(ToolError):
    """Cannot connect to backend store."""

    def __init__(self, store: str, error: str):
        super().__init__(
            f"Cannot connect to {store}. Please check that the service is running. "
            f"Error: {error}",
            {"store": store, "error": error},
        )


class EmbeddingError(ToolError):
    """Error generating embeddings."""

    def __init__(self, message: str, provider: str | None = None):
        details = {}
        if provider:
            details["provider"] = provider
        super().__init__(
            f"Embedding generation failed: {message}. "
            "Check embedding provider configuration and API keys.",
            details,
        )
