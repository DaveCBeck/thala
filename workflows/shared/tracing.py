"""Centralized tracing utilities for LangSmith integration.

This module provides consistent patterns for:
1. Decorating workflow entry points with @traceable
2. Propagating trace context to subgraphs
3. Adding dynamic metadata for filtering in LangSmith

Usage:
    from workflows.shared.tracing import workflow_traceable, get_trace_config

    @workflow_traceable(name="MyWorkflow", workflow_type="my_workflow")
    async def my_workflow(topic: str, quality: str = "standard") -> dict:
        # Add dynamic metadata at runtime
        add_trace_metadata({"quality_tier": quality, "topic": topic[:50]})

        # Invoke subgraphs with trace config for parent linking
        result = await subgraph.ainvoke(state, config=get_trace_config())
        return result
"""

from functools import wraps
from typing import Any, Callable, TypeVar

from langsmith import get_current_run_tree, traceable

F = TypeVar("F", bound=Callable[..., Any])


def workflow_traceable(name: str, workflow_type: str) -> Callable[[F], F]:
    """Decorator for workflow entry points.

    Creates a root trace for the workflow with consistent naming and tags.
    Enables filtering in LangSmith by workflow type.

    Args:
        name: Display name for the trace (e.g., "AcademicLitReview")
        workflow_type: Workflow identifier for filtering (e.g., "lit_review")

    Example:
        @workflow_traceable(name="AcademicLitReview", workflow_type="lit_review")
        async def academic_lit_review(topic: str, ...) -> dict:
            ...
    """
    return traceable(
        run_type="chain",
        name=name,
        tags=[f"workflow:{workflow_type}"],
    )


def get_trace_config() -> dict[str, Any]:
    """Get config dict for subgraph invocations that preserves parent trace.

    When invoking a LangGraph subgraph from within a @traceable function,
    call this to get a config dict that links the subgraph trace as a child
    of the current trace.

    Returns:
        Config dict with callbacks for parent linking, or empty dict if
        no current trace exists.

    Example:
        @workflow_traceable(name="MyWorkflow", workflow_type="my_workflow")
        async def my_workflow(...):
            # This subgraph will appear as a child of MyWorkflow trace
            result = await subgraph.ainvoke(state, config=get_trace_config())
    """
    config: dict[str, Any] = {}
    if run_tree := get_current_run_tree():
        config["callbacks"] = run_tree.get_child_callbacks()
    return config


def merge_trace_config(existing_config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge trace config with an existing config dict.

    Use this when you have an existing config (e.g., with recursion_limit)
    and want to add trace parent linking.

    Args:
        existing_config: Existing config dict or None

    Returns:
        Merged config with trace callbacks added

    Example:
        config = {"recursion_limit": 100}
        result = await graph.ainvoke(state, config=merge_trace_config(config))
    """
    trace_config = get_trace_config()
    if existing_config is None:
        return trace_config

    merged = dict(existing_config)
    if "callbacks" in trace_config:
        # Merge callbacks if both have them
        if "callbacks" in merged and merged["callbacks"]:
            if isinstance(merged["callbacks"], list):
                merged["callbacks"] = merged["callbacks"] + trace_config["callbacks"]
            else:
                merged["callbacks"] = [merged["callbacks"]] + trace_config["callbacks"]
        else:
            merged["callbacks"] = trace_config["callbacks"]
    return merged


def add_trace_metadata(metadata: dict[str, Any]) -> None:
    """Add dynamic metadata to the current trace for filtering in LangSmith.

    Call this within a @traceable function to add runtime metadata that
    can be used for filtering in LangSmith UI.

    Args:
        metadata: Key-value pairs to add to the trace

    Example:
        @workflow_traceable(name="AcademicLitReview", workflow_type="lit_review")
        async def academic_lit_review(topic: str, quality: str = "standard", ...):
            add_trace_metadata({
                "quality_tier": quality,
                "topic": topic[:50],  # Truncate for filtering
                "language": "en",
            })
            ...

    LangSmith queries:
        metadata_key = "quality_tier" AND metadata_value = "high_quality"
        metadata_key = "language" AND metadata_value = "es"
    """
    if run_tree := get_current_run_tree():
        run_tree.add_metadata(metadata)
