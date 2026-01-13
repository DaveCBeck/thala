"""
Multi-language research workflow.

Runs research across multiple languages, producing:
1. A synthesized document integrating all findings
2. A comparative document analyzing cross-language patterns

Supports pluggable workflows via the workflow registry.
"""

# Register built-in workflows before importing API (which uses registry)
from workflows.multi_lang.builtin_workflows import register_builtin_workflows

register_builtin_workflows()

from workflows.multi_lang.graph.api import multi_lang_research, MultiLangResult
from workflows.multi_lang.workflow_registry import (
    register_workflow,
    get_available_workflows,
)

__all__ = [
    "multi_lang_research",
    "MultiLangResult",
    "register_workflow",
    "get_available_workflows",
]
