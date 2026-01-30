"""
Task queue infrastructure for managing long-running workflows.

Provides:
- Persistent JSON-based task queue (LLM-editable)
- Flexible concurrency control (max concurrent or stagger-based)
- Checkpoint/resume capability
- Budget awareness via LangSmith cost aggregation
- Category-based round-robin scheduling

Currently supports "topic" tasks (literature review workflows).
Designed to be extensible for other task types.
"""

from .schemas import (
    TaskStatus,
    TaskCategory,
    TaskPriority,
    TopicTask,
    TaskQueue,
    ConcurrencyConfig,
    WorkflowCheckpoint,
    CurrentWork,
    CostEntry,
    CostCache,
)
from .queue_manager import TaskQueueManager
from .checkpoint import CheckpointManager
from .budget_tracker import BudgetTracker

__all__ = [
    # Schemas
    "TaskStatus",
    "TaskCategory",
    "TaskPriority",
    "TopicTask",
    "TaskQueue",
    "ConcurrencyConfig",
    "WorkflowCheckpoint",
    "CurrentWork",
    "CostEntry",
    "CostCache",
    # Managers
    "TaskQueueManager",
    "CheckpointManager",
    "BudgetTracker",
]
