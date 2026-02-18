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


class IllustratePublishManifest(TypedDict):
    """Schema for the manifest.json written by lit_review_full save_and_spawn phase.

    Makes the inter-workflow contract explicit and testable.
    """

    topic: str
    category: str
    quality: str
    source_task_id: str
    output_dir: str  # Absolute path to unillustrated directory
    articles: list[dict]  # [{id, title, filename}, ...]


class IllustratePublishItem(TypedDict):
    """A single article in an illustrate_and_publish task."""

    id: str  # "overview", "deep_dive_1", etc.
    title: str
    source_path: str  # Path to unillustrated markdown
    illustrated: bool  # Has illustration completed?
    illustrated_path: Optional[str]  # Path to illustrated markdown (once done)
    draft_id: Optional[str]  # Substack draft ID (once published)
    draft_url: Optional[str]  # Substack draft URL (once published)


class IllustrateAndPublishTask(TypedDict):
    """An illustrate-and-publish task in the queue.

    Budget-aware illustration + immediate Substack draft publishing.
    Spawned by lit_review_full after saving unillustrated articles.
    """

    id: str
    task_type: str  # "illustrate_and_publish"
    status: str
    category: str
    priority: int
    quality: str
    source_task_id: str  # Parent lit_review_full task ID
    topic: str
    manifest_path: str  # Path to manifest.json
    items: list[IllustratePublishItem]
    not_before: Optional[str]  # ISO datetime — invisible to dispatcher until this time
    next_run_after: Optional[str]  # ISO datetime for DEFERRED scheduling

    # Timestamps (ISO format)
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    # Workflow tracking
    langsmith_run_id: Optional[str]
    current_phase: Optional[str]
    error_message: Optional[str]

    # Metadata
    notes: Optional[str]
    tags: list[str]


# Union type for all task types
Task = Union[TopicTask, WebResearchTask, IllustrateAndPublishTask]
