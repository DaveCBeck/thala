"""Task TypedDict schemas for the task queue system."""

from __future__ import annotations

from typing_extensions import TypedDict


class TopicTask(TypedDict):
    """A topic task in the queue (full literature review workflow).

    This is the original task type, now with explicit task_type field.
    Defaults to "lit_review_full" for backward compatibility.
    """

    id: str  # UUID
    task_type: str  # "lit_review_full" (default for backward compat)
    topic: str  # Main topic text
    research_questions: list[str] | None  # Optional pre-defined questions
    category: str  # TaskCategory value
    priority: int  # TaskPriority value (1-4)
    status: str  # TaskStatus value
    quality: str  # "quick", "standard", etc.
    language: str  # ISO 639-1 code
    date_range: tuple[int, int] | None  # (start_year, end_year)

    # Timestamps (ISO format)
    created_at: str  # When added to queue
    started_at: str | None  # When workflow began
    completed_at: str | None  # When workflow finished

    # Workflow tracking
    langsmith_run_id: str | None  # For cost attribution and trace lookup
    current_phase: str | None  # Last checkpoint phase
    error_message: str | None  # If failed

    # Metadata for LLM editing
    notes: str | None  # User/LLM notes
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
    language: str | None  # ISO 639-1 code (optional for web research)

    # Timestamps (ISO format)
    created_at: str  # When added to queue
    started_at: str | None  # When workflow began
    completed_at: str | None  # When workflow finished

    # Workflow tracking
    langsmith_run_id: str | None  # For cost attribution and trace lookup
    current_phase: str | None  # Last checkpoint phase
    error_message: str | None  # If failed

    # Metadata for LLM editing
    notes: str | None  # User/LLM notes
    tags: list[str]  # Searchable tags


class IllustrateExportManifest(TypedDict):
    """Schema for the manifest.json written by lit_review_full save_and_spawn phase.

    Makes the inter-workflow contract explicit and testable.
    Used by the illustrate_and_export workflow to locate source articles.
    """

    topic: str
    category: str
    quality: str
    source_task_id: str
    output_dir: str  # Absolute path to unillustrated directory
    articles: list[dict]  # [{id, title, filename}, ...]


class IllustrateExportItem(TypedDict):
    """A single article in an illustrate_and_export task."""

    id: str  # "overview", "deep_dive_1", etc.
    title: str
    subtitle: str  # Short subtitle for Substack draft
    source_path: str  # Path to unillustrated markdown
    illustrated: bool  # Has illustration completed?
    illustrated_path: str | None  # Path to illustrated markdown (once done)
    exported: bool  # Has the article been exported to a batch folder?


class IllustrateAndExportTask(TypedDict):
    """An illustrate-and-export task in the queue.

    Budget-aware illustration + batch export to VPS via rsync.
    Spawned by lit_review_full after saving unillustrated articles.
    """

    id: str
    task_type: str  # "illustrate_and_export"
    status: str
    category: str
    priority: int
    quality: str
    source_task_id: str  # Parent lit_review_full task ID
    topic: str
    manifest_path: str  # Path to manifest.json
    items: list[IllustrateExportItem]
    not_before: str | None  # ISO datetime — invisible to dispatcher until this time
    next_run_after: str | None  # ISO datetime for DEFERRED scheduling

    # Timestamps (ISO format)
    created_at: str
    started_at: str | None
    completed_at: str | None

    # Workflow tracking
    langsmith_run_id: str | None
    current_phase: str | None
    error_message: str | None

    # Metadata
    notes: str | None
    tags: list[str]


# Union type for all task types
Task = TopicTask | WebResearchTask | IllustrateAndExportTask
