"""Shared modules for supervision loops.

Contains types, prompts, nodes, and utilities used by Loop 1 (theoretical depth)
and Loop 2 (literature expansion) in the enhancement supervision workflow.
"""

from workflows.enhance.supervision.shared.types import (
    IdentifiedIssue,
    SupervisorDecision,
    LiteratureBase,
    LiteratureBaseDecision,
)
from workflows.enhance.supervision.shared.routing import (
    route_after_analysis,
    should_continue_supervision,
)
from workflows.enhance.supervision.shared.nodes import (
    analyze_review_node,
    expand_topic_node,
    integrate_content_node,
)
from workflows.enhance.supervision.shared.focused_expansion import run_focused_expansion
from workflows.enhance.supervision.shared.mini_review import run_mini_review

__all__ = [
    # Types
    "IdentifiedIssue",
    "SupervisorDecision",
    "LiteratureBase",
    "LiteratureBaseDecision",
    # Routing
    "route_after_analysis",
    "should_continue_supervision",
    # Nodes
    "analyze_review_node",
    "expand_topic_node",
    "integrate_content_node",
    # Utilities
    "run_focused_expansion",
    "run_mini_review",
]
