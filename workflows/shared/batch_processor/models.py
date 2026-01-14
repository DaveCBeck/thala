"""Data models for batch processing."""

import re
from dataclasses import dataclass
from typing import Optional


def sanitize_custom_id(identifier: str) -> str:
    """Convert identifier to valid Anthropic batch custom_id.

    The API requires custom_id to match pattern ^[a-zA-Z0-9_-]{1,64}$.
    This replaces any invalid character with underscore and truncates to 64 chars.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
    return sanitized[:64]


@dataclass
class BatchRequest:
    """A single request to be included in a batch."""

    custom_id: str
    prompt: str
    model: any  # ModelTier
    max_tokens: int = 4096
    system: Optional[str] = None
    thinking_budget: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None


@dataclass
class BatchResult:
    """Result from a single batch request."""

    custom_id: str
    success: bool
    content: Optional[str] = None
    thinking: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None
