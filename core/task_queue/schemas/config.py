"""Configuration TypedDict schemas for the task queue system."""

from typing_extensions import TypedDict

from .tasks import Task


class TaskQueue(TypedDict):
    """Root queue structure (queue.json).

    Version 2.0: Two separate arrays for research and publish tasks.
    Research tasks use category round-robin selection.
    Publish tasks use date-gated priority selection.
    """

    version: str  # Schema version ("2.0")
    categories: list[str]  # Category names for research round-robin
    last_category_index: int  # For round-robin tracking
    research_tasks: list[Task]  # lit_review_full, web_research
    publish_tasks: list[Task]  # illustrate_and_export
    last_updated: str  # ISO datetime
