"""
TypedDict schemas for the task queue system.

Task is the base concept. TopicTask is the current implementation
for literature review workflows. The system is designed to support
other task types in the future.
"""

from enum import Enum
from typing import Literal, Optional

from typing_extensions import TypedDict


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"  # Not yet started
    IN_PROGRESS = "in_progress"  # Currently running
    PAUSED = "paused"  # Manually paused or budget-paused
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed with error


class TaskCategory(Enum):
    """Thematic categories for round-robin selection.

    These are placeholders - edit queue.json to customize.
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


class TopicTask(TypedDict):
    """A topic task in the queue (literature review workflow)."""

    id: str  # UUID
    topic: str  # Main topic text
    research_questions: Optional[list[str]]  # Optional pre-defined questions
    category: str  # TaskCategory value
    priority: int  # TaskPriority value (1-4)
    status: str  # TaskStatus value
    quality: str  # "quick", "standard", etc.
    language: str  # ISO 639-1 code
    date_range: Optional[tuple[int, int]]  # (start_year, end_year)

    # Timestamps (ISO format)
    created_at: str  # When added to queue
    started_at: Optional[str]  # When workflow began
    completed_at: Optional[str]  # When workflow finished

    # Workflow tracking
    langsmith_run_id: Optional[str]  # For cost attribution and trace lookup
    current_phase: Optional[str]  # Last checkpoint phase
    error_message: Optional[str]  # If failed

    # Metadata for LLM editing
    notes: Optional[str]  # User/LLM notes
    tags: list[str]  # Searchable tags


class ConcurrencyConfig(TypedDict):
    """Concurrency control configuration."""

    mode: Literal["max_concurrent", "stagger_hours"]
    max_concurrent: int  # Max simultaneous tasks (mode: max_concurrent)
    stagger_hours: float  # Hours between starts (mode: stagger_hours)


class TaskQueue(TypedDict):
    """Root queue structure (queue.json)."""

    version: str  # Schema version
    concurrency: ConcurrencyConfig
    categories: list[str]  # Category names for round-robin
    last_category_index: int  # For round-robin tracking
    topics: list[TopicTask]  # Topic tasks in queue
    last_updated: str  # ISO datetime


class WorkflowCheckpoint(TypedDict):
    """Checkpoint data for workflow resumption."""

    topic_id: str
    langsmith_run_id: str
    phase: str  # "discovery", "diffusion", "processing", "clustering", "synthesis"
    phase_progress: dict  # Phase-specific progress data
    started_at: str
    last_checkpoint_at: str

    # Phase-specific counters
    papers_discovered: int
    papers_processed: int
    diffusion_stage: int
    clusters_generated: bool
    synthesis_complete: bool
    supervision_loop: Optional[int]


class CurrentWork(TypedDict):
    """Active work state for resume capability (current_work.json)."""

    version: str
    active_topics: list[WorkflowCheckpoint]  # Topics currently being processed
    process_locks: dict[str, str]  # topic_id -> PID string


class CostEntry(TypedDict):
    """Cached cost data for a time period."""

    period: str  # "2026-01" (monthly)
    total_cost_usd: float
    token_breakdown: dict[str, int]  # model_name -> tokens
    run_count: int
    last_aggregated: str  # ISO datetime
    runs_included: list[str]  # Last N run_ids for reference


class CostCache(TypedDict):
    """Cost tracking cache (cost_cache.json)."""

    version: str
    periods: dict[str, CostEntry]  # period -> CostEntry
    last_sync: Optional[str]  # Last full LangSmith sync
