"""Enum types for the task queue system."""

from enum import Enum


class TaskType(Enum):
    """Workflow type discriminator.

    Each task type maps to a workflow implementation in core/task_queue/workflows/.
    """

    LIT_REVIEW_FULL = "lit_review_full"  # lit_review → enhance → evening_reads → save_and_spawn
    WEB_RESEARCH = "web_research"  # deep_research → evening_reads
    ILLUSTRATE_AND_EXPORT = "illustrate_and_export"  # Budget-aware illustration + batch export to VPS


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"  # Not yet started
    IN_PROGRESS = "in_progress"  # Currently running
    DEFERRED = "deferred"  # Re-runnable after next_run_after
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error


class TaskCategory(Enum):
    """Thematic categories for round-robin selection.

    NOTE: The source of truth for categories is .thala/queue/publications.json.
    This enum provides type hints but categories are loaded dynamically from
    publications.json at runtime. To add/remove categories, edit publications.json.
    """

    PHILOSOPHY = "philosophy"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    SOCIETY = "society"
    CULTURE = "culture"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
