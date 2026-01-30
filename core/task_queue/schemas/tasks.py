"""Task TypedDict schemas for the task queue system."""

from typing import Optional, Union

from typing_extensions import TypedDict


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
