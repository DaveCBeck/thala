"""
Workflow registry for pluggable workflow orchestration.

This module provides a registration system that allows workflows to be
dynamically registered and discovered without modifying orchestration logic.

Usage:
    from workflows.shared.wrappers import register_workflow, WorkflowResult

    async def my_workflow_adapter(
        query: str,
        quality: str,
        language: str,
        research_questions: list[str] | None,
        **kwargs,
    ) -> WorkflowResult:
        result = await my_workflow(...)
        return WorkflowResult(
            final_report=result.get("final_report"),
            source_count=result.get("count", 0),
            status=result.get("status", "success"),
            errors=result.get("errors", []),
        )

    register_workflow(
        key="my_workflow",
        name="My Workflow",
        runner=my_workflow_adapter,
        default_enabled=False,
        requires_questions=True,
        supports_date_range=False,
        description="Description of what this workflow does",
    )
"""

import logging
from typing import Callable, Awaitable
from typing_extensions import TypedDict

from workflows.shared.wrappers.result_types import WorkflowResult

logger = logging.getLogger(__name__)


class WorkflowConfig(TypedDict):
    """Configuration for a registered workflow."""

    name: str  # Display name for logging/UI
    runner: Callable[..., Awaitable[WorkflowResult]]  # Async adapter function
    default_enabled: bool  # Whether enabled by default
    requires_questions: bool  # Whether research_questions are needed
    supports_date_range: bool  # Whether date_range is accepted
    quality_tiers: list[str]  # Available quality tiers for this workflow
    description: str  # Human-readable description


# Global registry - populated by register_workflow calls
WORKFLOW_REGISTRY: dict[str, WorkflowConfig] = {}


def register_workflow(
    key: str,
    name: str,
    runner: Callable[..., Awaitable[WorkflowResult]],
    default_enabled: bool = False,
    requires_questions: bool = False,
    supports_date_range: bool = False,
    quality_tiers: list[str] | None = None,
    description: str = "",
) -> None:
    """Register a workflow for use in orchestration.

    Args:
        key: Unique identifier (e.g., "web_research", "academic_lit_review")
        name: Human-readable display name
        runner: Async function with signature:
            async def runner(
                query: str,
                quality: str,
                language: str = "en",
                research_questions: list[str] | None = None,
                **kwargs,
            ) -> WorkflowResult
        default_enabled: Whether this workflow runs by default
        requires_questions: Whether research_questions are required
        supports_date_range: Whether date_range parameter is supported
        quality_tiers: List of quality tiers supported (default: quick/standard/comprehensive)
        description: Human-readable description
    """
    if key in WORKFLOW_REGISTRY:
        logger.debug(f"Overwriting existing workflow registration: {key}")

    WORKFLOW_REGISTRY[key] = WorkflowConfig(
        name=name,
        runner=runner,
        default_enabled=default_enabled,
        requires_questions=requires_questions,
        supports_date_range=supports_date_range,
        quality_tiers=quality_tiers or ["quick", "standard", "comprehensive"],
        description=description,
    )
    logger.debug(f"Registered workflow: {key} ({name})")


def unregister_workflow(key: str) -> bool:
    """Remove a workflow from the registry.

    Args:
        key: The workflow key to remove

    Returns:
        True if removed, False if not found
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
                logger.debug(f"Unknown workflow in selection: {key}")
    return selection
