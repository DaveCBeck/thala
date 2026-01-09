"""
Workflow registry for pluggable multi-language research workflows.

This module re-exports the shared workflow registry infrastructure.
The actual registry implementation is in workflows.shared.wrappers.

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

# Re-export from shared wrappers for backwards compatibility
# The actual registry is shared across all wrapper workflows
from workflows.shared.wrappers.registry import (
    WorkflowConfig,
    WORKFLOW_REGISTRY,
    register_workflow,
    unregister_workflow,
    get_workflow,
    get_available_workflows,
    get_default_workflow_selection,
    build_workflow_selection,
)
from workflows.shared.wrappers.result_types import WorkflowResult

__all__ = [
    "WorkflowResult",
    "WorkflowConfig",
    "WORKFLOW_REGISTRY",
    "register_workflow",
    "unregister_workflow",
    "get_workflow",
    "get_available_workflows",
    "get_default_workflow_selection",
    "build_workflow_selection",
]
