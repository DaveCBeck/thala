"""
Shared workflow wrapper infrastructure.

This module provides common infrastructure for workflow orchestration:
- Registry: Dynamic workflow registration and discovery
- Quality: Unified quality tier system (QualityTier)
- Invoker: Standardized workflow invocation
- Result types: Common result schemas

Usage:
    from workflows.shared.wrappers import (
        register_workflow,
        invoke_workflow,
        WorkflowResult,
    )
"""

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
from workflows.shared.wrappers.quality import (
    get_quality_tiers,
)
from workflows.shared.wrappers.result_types import (
    WorkflowResult,
)
from workflows.shared.wrappers.invoker import (
    invoke_workflow,
)

__all__ = [
    # Registry
    "WorkflowConfig",
    "WORKFLOW_REGISTRY",
    "register_workflow",
    "unregister_workflow",
    "get_workflow",
    "get_available_workflows",
    "get_default_workflow_selection",
    "build_workflow_selection",
    # Quality
    "get_quality_tiers",
    # Result types
    "WorkflowResult",
    # Invoker
    "invoke_workflow",
]
