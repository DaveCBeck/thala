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

# Re-export all public symbols for backward compatibility
from .callbacks import IncrementalCheckpointCallback, PhaseCheckpointCallback
from .checkpoints import CurrentWork, WorkflowCheckpoint
from .config import ConcurrencyConfig, TaskQueue
from .cost import CostCache, CostEntry, IncrementalState
from .enums import TaskCategory, TaskPriority, TaskStatus, TaskType
from .tasks import PublishItem, PublishSeriesTask, Task, TopicTask, WebResearchTask

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
    "PublishItem",
    "PublishSeriesTask",
    "Task",
    # Config
    "ConcurrencyConfig",
    "TaskQueue",
    # Checkpoints
    "WorkflowCheckpoint",
    "CurrentWork",
    # Cost
    "CostEntry",
    "CostCache",
    "IncrementalState",
]
