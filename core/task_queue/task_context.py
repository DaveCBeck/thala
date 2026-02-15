"""Task context propagation via ContextVar.

Provides task-level metadata (task_id, task_type, topic, quality_tier)
to all workflow entry points via a ContextVar. Set by workflow_executor,
read by workflow ainvoke() calls to attach shared tracing metadata.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskContext:
    """Immutable snapshot of the current task's identity."""

    task_id: str
    task_type: str
    topic: str
    quality_tier: str


_task_context_var: ContextVar[TaskContext | None] = ContextVar("task_context", default=None)


def set_task_context(task_id: str, task_type: str, topic: str, quality_tier: str) -> None:
    """Set the current task context (called by workflow_executor)."""
    _task_context_var.set(TaskContext(task_id, task_type, topic, quality_tier))


def get_task_context() -> TaskContext | None:
    """Get the current task context, or None if not in a task."""
    return _task_context_var.get()


def clear_task_context() -> None:
    """Clear the current task context (called in workflow_executor finally block)."""
    _task_context_var.set(None)


def get_trace_metadata() -> dict[str, str]:
    """Return task metadata dict for LangSmith tracing, or {} if no context."""
    ctx = _task_context_var.get()
    if ctx is None:
        return {}
    return {
        "task_id": ctx.task_id,
        "task_type": ctx.task_type,
        "topic": ctx.topic[:100],
        "quality_tier": ctx.quality_tier,
    }


def get_trace_tags() -> list[str]:
    """Return task tags for LangSmith tracing, or [] if no context."""
    ctx = _task_context_var.get()
    if ctx is None:
        return []
    return [f"task:{ctx.task_id}", f"type:{ctx.task_type}"]
