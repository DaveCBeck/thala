"""Configuration TypedDict schemas for the task queue system."""

from typing import Literal

from typing_extensions import TypedDict

from .tasks import Task


class ConcurrencyConfig(TypedDict):
    """Concurrency control configuration."""

    mode: Literal["max_concurrent", "stagger_hours"]
    max_concurrent: int  # Max simultaneous tasks (mode: max_concurrent)
    stagger_hours: float  # Hours between starts (mode: stagger_hours)


class TaskQueue(TypedDict):
    """Root queue structure (queue.json).

    The 'topics' field name is kept for backward compatibility but
    can contain any task type (TopicTask, WebResearchTask, etc.).
    """

    version: str  # Schema version
    concurrency: ConcurrencyConfig
    categories: list[str]  # Category names for round-robin
    last_category_index: int  # For round-robin tracking
    topics: list[Task]  # Tasks in queue (any type)
    last_updated: str  # ISO datetime
