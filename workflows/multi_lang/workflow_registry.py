"""
Workflow registry for pluggable multi-language research workflows.

This module provides a registration system that allows workflows to be
dynamically added to the multi_lang orchestration without modifying
core dispatch logic.

Usage:
    from workflows.multi_lang.workflow_registry import register_workflow

    async def my_workflow_adapter(topic, language_config, quality, research_questions):
        result = await my_workflow(...)
        return {
            "final_report": result.get("report"),
            "source_count": result.get("count", 0),
            "status": "completed" if result.get("report") else "failed",
            "errors": result.get("errors", []),
        }

    register_workflow(
        key="my_workflow",
        name="My Workflow",
        runner=my_workflow_adapter,
        default_enabled=False,
        requires_questions=True,
        description="Description of what this workflow does",
    )
"""

import logging
from typing import Callable, Awaitable, Any
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class WorkflowResult(TypedDict):
    """Standard result format all workflow adapters must return."""

    final_report: str | None  # The main text/markdown output
    source_count: int  # Number of sources found/processed
    status: str  # "completed" | "failed"
    errors: list[dict]  # List of {phase, error} dicts


class WorkflowConfig(TypedDict):
    """Configuration for a registered workflow."""

    name: str  # Display name for logging/UI
    runner: Callable[..., Awaitable[WorkflowResult]]  # Async adapter function
    default_enabled: bool  # Whether enabled by default in WorkflowSelection
    requires_questions: bool  # Whether research_questions are passed to runner
    description: str  # Human-readable description


# Global registry - populated by register_workflow calls
WORKFLOW_REGISTRY: dict[str, WorkflowConfig] = {}


def register_workflow(
    key: str,
    name: str,
    runner: Callable[..., Awaitable[WorkflowResult]],
    default_enabled: bool = False,
    requires_questions: bool = True,
    description: str = "",
) -> None:
    """Register a workflow for use in multi_lang orchestration.

    Args:
        key: Unique identifier for the workflow (e.g., "web", "academic", "books")
        name: Human-readable display name
        runner: Async function with signature:
            async def runner(
                topic: str,
                language_config: dict,
                quality: str,
                research_questions: list[str] | None,  # Only if requires_questions=True
            ) -> WorkflowResult
        default_enabled: Whether this workflow runs by default
        requires_questions: Whether to pass research_questions to the runner
        description: Human-readable description of what the workflow does
    """
    if key in WORKFLOW_REGISTRY:
        logger.warning(f"Overwriting existing workflow registration: {key}")

    WORKFLOW_REGISTRY[key] = WorkflowConfig(
        name=name,
        runner=runner,
        default_enabled=default_enabled,
        requires_questions=requires_questions,
        description=description,
    )
    logger.debug(f"Registered workflow: {key} ({name})")


def unregister_workflow(key: str) -> bool:
    """Remove a workflow from the registry.

    Args:
        key: The workflow key to remove

    Returns:
        True if the workflow was removed, False if it wasn't registered
    """
    if key in WORKFLOW_REGISTRY:
        del WORKFLOW_REGISTRY[key]
        logger.debug(f"Unregistered workflow: {key}")
        return True
    return False


def get_workflow(key: str) -> WorkflowConfig | None:
    """Get a workflow configuration by key.

    Args:
        key: The workflow key to look up

    Returns:
        The WorkflowConfig if found, None otherwise
    """
    return WORKFLOW_REGISTRY.get(key)


def get_available_workflows() -> list[str]:
    """Return list of registered workflow keys."""
    return list(WORKFLOW_REGISTRY.keys())


def get_default_workflow_selection() -> dict[str, bool]:
    """Build default workflow selection from registry.

    Returns:
        Dict mapping workflow key to default_enabled value
    """
    return {key: config["default_enabled"] for key, config in WORKFLOW_REGISTRY.items()}


def build_workflow_selection(
    user_selection: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Build workflow selection, applying user overrides to defaults.

    Args:
        user_selection: Optional user-provided workflow selection dict

    Returns:
        Complete workflow selection with all registered workflows
    """
    selection = get_default_workflow_selection()
    if user_selection:
        for key, enabled in user_selection.items():
            if key in WORKFLOW_REGISTRY:
                selection[key] = enabled
            else:
                logger.warning(f"Unknown workflow in selection: {key}")
    return selection
