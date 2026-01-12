"""Loop 3: Structure and Cohesion - Two-phase structural editing.

This loop uses a two-phase analyst pattern:
- Phase A: Identify structural issues (diagnosis)
- Phase B: Generate concrete edits based on identified issues (prescription)

This separation gives the LLM a clear cognitive path and eliminates the
"identifies issues but returns empty edits" failure mode.
"""

from .graph import Loop3State, create_loop3_graph, run_loop3_standalone
from .analyzer import (
    analyze_structure_phase_a_node,
    generate_edits_phase_b_node,
    validate_issue_edit_mapping,
)
from .fallback import retry_analyze_node, execute_manifest_node
from .validator import (
    validate_edits_node,
    apply_edits_programmatically_node,
    verify_application_node,
)
from .verification import verify_architecture_node
from .routing import (
    route_after_phase_a,
    route_after_analysis,
    route_after_validation,
    check_continue,
)
from .utils import (
    number_paragraphs_node,
    validate_result_node,
    increment_iteration,
    finalize_node,
)

__all__ = [
    "Loop3State",
    "create_loop3_graph",
    "run_loop3_standalone",
    "analyze_structure_phase_a_node",
    "generate_edits_phase_b_node",
    "validate_issue_edit_mapping",
    "retry_analyze_node",
    "execute_manifest_node",
    "validate_edits_node",
    "apply_edits_programmatically_node",
    "verify_application_node",
    "verify_architecture_node",
    "route_after_phase_a",
    "route_after_analysis",
    "route_after_validation",
    "check_continue",
    "number_paragraphs_node",
    "validate_result_node",
    "increment_iteration",
    "finalize_node",
]
