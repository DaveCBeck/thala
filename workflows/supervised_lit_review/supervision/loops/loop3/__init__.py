"""Loop 3: Structure and Cohesion - Section-rewrite based structural editing.

This loop uses a two-phase approach:
- Phase A: Identify structural issues (diagnosis)
- Phase B: Rewrite affected sections to fix issues (prescription)

The section-rewrite approach (new) replaces the previous edit-specification approach.
Instead of generating structured edit operations, we directly rewrite the affected
sections. This eliminates the "identifies issues but can't generate valid edits"
failure mode.
"""

from .graph import Loop3State, create_loop3_graph, run_loop3_standalone
from .analyzer import (
    analyze_structure_phase_a_node,
    # Legacy exports for backward compatibility
    generate_edits_phase_b_node,
    validate_issue_edit_mapping,
)
from .section_rewriter import rewrite_sections_for_issues_node
from .verification import verify_architecture_node
from .routing import (
    route_after_phase_a,
    check_continue_rewrite,
    # Legacy exports for backward compatibility
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
# Legacy imports for backward compatibility
from .fallback import retry_analyze_node, execute_manifest_node
from .validator import (
    validate_edits_node,
    apply_edits_programmatically_node,
    verify_application_node,
)

__all__ = [
    # Core graph
    "Loop3State",
    "create_loop3_graph",
    "run_loop3_standalone",
    # Phase A (unchanged)
    "analyze_structure_phase_a_node",
    # Phase B (new - section rewriting)
    "rewrite_sections_for_issues_node",
    # Verification
    "verify_architecture_node",
    # Routing
    "route_after_phase_a",
    "check_continue_rewrite",
    # Utilities
    "number_paragraphs_node",
    "validate_result_node",
    "increment_iteration",
    "finalize_node",
    # Legacy exports (kept for backward compatibility)
    "generate_edits_phase_b_node",
    "validate_issue_edit_mapping",
    "retry_analyze_node",
    "execute_manifest_node",
    "validate_edits_node",
    "apply_edits_programmatically_node",
    "verify_application_node",
    "route_after_analysis",
    "route_after_validation",
    "check_continue",
]
