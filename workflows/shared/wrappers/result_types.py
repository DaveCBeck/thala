"""
Shared result types for workflow wrappers.

These types define the standard interface for workflow results.
"""

from typing_extensions import TypedDict


class WorkflowResult(TypedDict):
    """Standard result format for workflow adapters in registry-based invocation.

    This is the minimal result type returned by workflow adapters.
    Used by multi_lang workflow registry.
    """

    final_report: str | None  # The main text/markdown output
    source_count: int  # Number of sources found/processed
    status: str  # "success" | "partial" | "failed"
    errors: list[dict]  # List of {phase, error} dicts
