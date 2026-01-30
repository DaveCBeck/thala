"""
TypedDict schemas for the task queue system.

Supports multiple task types via discriminated union pattern:
- TopicTask (task_type="lit_review_full"): Academic literature review workflow
- WebResearchTask (task_type="web_research"): Web research workflow

To add a new task type:
1. Create a new TypedDict here with required fields
2. Add to the Task union type
3. Implement workflow in core/task_queue/workflows/
"""

from enum import Enum
from typing import Literal, Optional, Union

from typing_extensions import TypedDict


class TaskType(Enum):
    """Workflow type discriminator.

    Each task type maps to a workflow implementation in core/task_queue/workflows/.
    """

    LIT_REVIEW_FULL = "lit_review_full"  # lit_review → enhance → evening_reads → illustrate
    WEB_RESEARCH = "web_research"  # deep_research → evening_reads
    PUBLISH_SERIES = "publish_series"  # Schedule-aware draft publishing


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"  # Not yet started
    IN_PROGRESS = "in_progress"  # Currently running
    PAUSED = "paused"  # Manually paused or budget-paused
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


class TopicTask(TypedDict):
    """A topic task in the queue (full literature review workflow).

    This is the original task type, now with explicit task_type field.
    Defaults to "lit_review_full" for backward compatibility.
    """

    id: str  # UUID
    task_type: str  # "lit_review_full" (default for backward compat)
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


class WebResearchTask(TypedDict):
    """A web research task in the queue.

    Uses deep_research workflow followed by evening_reads.
    """

    id: str  # UUID
    task_type: str  # "web_research"
    query: str  # Research query
    category: str  # TaskCategory value
    priority: int  # TaskPriority value (1-4)
    status: str  # TaskStatus value
    quality: str  # "quick", "standard", etc.
    language: Optional[str]  # ISO 639-1 code (optional for web research)

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


class PublishItem(TypedDict):
    """A single item in a publish series."""

    id: str  # "overview", "lit_review", "deep_dive_1", etc.
    title: str  # Article title
    path: str  # Path to illustrated markdown file
    day_offset: int  # Days from base_date to publish
    audience: str  # "everyone" or "only_paid"
    published: bool  # Has this item been published?
    draft_id: Optional[str]  # Substack draft ID once created
    draft_url: Optional[str]  # URL to draft in Substack


class PublishSeriesTask(TypedDict):
    """A publish series task in the queue.

    Schedule-aware draft publishing via Substack API.
    Spawned by lit_review_full after illustration completes.
    """

    id: str  # UUID
    task_type: str  # "publish_series"
    category: str  # For publication routing
    priority: int  # TaskPriority value (1-4)
    status: str  # TaskStatus value
    quality: str  # Inherited from parent task

    # Publish-specific fields
    base_date: str  # ISO datetime (Monday 3pm local)
    items: list[PublishItem]  # The 5 items to publish
    source_task_id: str  # ID of lit_review_full task that spawned this

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


# Union type for all task types
Task = Union[TopicTask, WebResearchTask, PublishSeriesTask]


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


class WorkflowCheckpoint(TypedDict):
    """Checkpoint data for workflow resumption.

    Now workflow-aware: task_type determines valid phases.
    Stores phase_outputs for resumption after interruption.
    """

    task_id: str  # Renamed from topic_id for genericity
    task_type: str  # Workflow type for phase validation
    langsmith_run_id: str
    phase: str  # Current phase (workflow-specific)
    phase_progress: dict  # Phase-specific progress data
    phase_outputs: dict  # Outputs from completed phases for resumption
    started_at: str
    last_checkpoint_at: str

    # Generic counters storage (workflow-specific)
    # For lit_review_full: papers_discovered, papers_processed, etc.
    # For web_research: sources_found, etc.
    counters: dict  # {counter_name: value}


class CurrentWork(TypedDict):
    """Active work state for resume capability (current_work.json)."""

    version: str
    active_tasks: list[WorkflowCheckpoint]  # Tasks currently being processed
    process_locks: dict[str, str]  # task_id -> PID string


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


class IncrementalState(TypedDict):
    """Incremental checkpoint state for mid-phase resumption.

    Stored at .thala/queue/incremental/{task_id}.json
    Allows resuming iterative phases (paper processing, supervision loops)
    from the last checkpoint rather than restarting the entire phase.
    """

    task_id: str
    phase: str
    iteration_count: int  # Number of items processed
    checkpoint_interval: int  # Every N items (for reference)
    partial_results: dict  # Keyed by identifier (DOI, loop_id, etc.)
    last_checkpoint_at: str  # ISO timestamp
