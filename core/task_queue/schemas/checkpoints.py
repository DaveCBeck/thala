"""Checkpoint TypedDict schemas for the task queue system."""

from typing_extensions import TypedDict


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
