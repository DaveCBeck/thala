"""Workflow registry for task queue.

Provides a registry pattern for workflow implementations, allowing
different task types to use different workflow pipelines.

Usage:
    from core.task_queue.workflows import get_workflow, get_phases, WORKFLOW_REGISTRY

    # Get workflow instance
    workflow = get_workflow("lit_review_full")

    # Run workflow
    result = await workflow.run(task, checkpoint_callback)

    # Get phases for a workflow type
    phases = get_phases("web_research")
"""

from typing import TYPE_CHECKING

from .base import BaseWorkflow

if TYPE_CHECKING:
    pass

# Import workflow implementations
from .illustrate_and_export import IllustrateAndExportWorkflow
from .lit_review_full import LitReviewFullWorkflow
from .lit_review_web_augmented import LitReviewWebAugmentedWorkflow
from .web_research import WebResearchWorkflow

# Registry mapping task_type -> workflow class
WORKFLOW_REGISTRY: dict[str, type[BaseWorkflow]] = {
    "lit_review_full": LitReviewFullWorkflow,
    "lit_review_web_augmented": LitReviewWebAugmentedWorkflow,
    "web_research": WebResearchWorkflow,
    "illustrate_and_export": IllustrateAndExportWorkflow,
}

# Default workflow type for backward compatibility
DEFAULT_WORKFLOW_TYPE = "lit_review_full"


def get_workflow(task_type: str) -> BaseWorkflow:
    """Get workflow instance for a task type.

    Args:
        task_type: The workflow type identifier (e.g., "lit_review_full", "web_research")

    Returns:
        Instantiated workflow object

    Raises:
        ValueError: If task_type is not registered
    """
    if task_type not in WORKFLOW_REGISTRY:
        available = ", ".join(WORKFLOW_REGISTRY.keys())
        raise ValueError(f"Unknown task type: {task_type}. Available: {available}")
    return WORKFLOW_REGISTRY[task_type]()


def get_phases(task_type: str) -> list[str]:
    """Get checkpoint phases for a workflow type."""
    return get_workflow(task_type).phases


def get_available_types() -> list[str]:
    """Get list of available workflow types."""
    return list(WORKFLOW_REGISTRY.keys())


__all__ = [
    "BaseWorkflow",
    "WORKFLOW_REGISTRY",
    "DEFAULT_WORKFLOW_TYPE",
    "get_workflow",
    "get_phases",
    "get_available_types",
    "IllustrateAndExportWorkflow",
    "LitReviewFullWorkflow",
    "LitReviewWebAugmentedWorkflow",
    "WebResearchWorkflow",
]
