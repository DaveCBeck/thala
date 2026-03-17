"""
TypedDict schemas for the task queue system.

Supports multiple task types via discriminated union pattern:
- TopicTask (task_type="lit_review_full"): Academic literature review workflow
- WebResearchTask (task_type="web_research"): Web research workflow
- IllustrateAndExportTask (task_type="illustrate_and_export"): Illustration + batch export to VPS

To add a new task type:
1. Create a new TypedDict here with required fields
2. Add to the Task union type
3. Implement workflow in core/task_queue/workflows/
"""

# Re-export all public symbols for backward compatibility
from .callbacks import IncrementalCheckpointCallback, PhaseCheckpointCallback
from .checkpoints import CurrentWork, WorkflowCheckpoint
from .config import TaskQueue
from .cost import CostCache, CostEntry, IncrementalState
from .enums import TaskCategory, TaskPriority, TaskStatus, TaskType
from .tasks import (
    IllustrateAndExportTask,
    IllustrateExportItem,
    IllustrateExportManifest,
    Task,
    TopicTask,
    WebResearchTask,
)

__all__ = [
    # Callbacks
    "PhaseCheckpointCallback",
    "IncrementalCheckpointCallback",
    # Enums
    "TaskType",
    "TaskStatus",
    "TaskCategory",
    "TaskPriority",
    # Tasks
    "TopicTask",
    "WebResearchTask",
    "IllustrateExportManifest",
    "IllustrateExportItem",
    "IllustrateAndExportTask",
    "Task",
    # Config
    "TaskQueue",
    # Checkpoints
    "WorkflowCheckpoint",
    "CurrentWork",
    # Cost
    "CostEntry",
    "CostCache",
    "IncrementalState",
]
