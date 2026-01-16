"""Node implementations for the editing workflow."""

from .parse_document import parse_document_node
from .analyze_structure import analyze_structure_node
from .plan_edits import plan_edits_node
from .execute_edits import (
    route_to_edit_workers,
    execute_structure_edits_worker,
    execute_generation_edit_worker,
    execute_removal_edit_worker,
    assemble_edits_node,
)
from .verify_structure import verify_structure_node, check_structure_complete
from .detect_citations import detect_citations_node, route_to_enhance_or_polish
from .enhance_section import (
    route_to_enhance_sections,
    enhance_section_worker,
    assemble_enhancements_node,
)
from .enhance_coherence import enhance_coherence_review_node, route_enhance_iteration
from .fact_check import (
    screen_sections_for_fact_check,
    route_to_fact_check_sections,
    fact_check_section_worker,
    assemble_fact_checks_node,
)
from .reference_check import (
    pre_validate_citations,
    route_to_reference_check_sections,
    reference_check_section_worker,
    assemble_reference_checks_node,
)
from .apply_verified_edits import apply_verified_edits_node
from .polish import polish_node
from .finalize import finalize_node

__all__ = [
    # Phase 1: Parse
    "parse_document_node",
    # Phase 2: Analyze
    "analyze_structure_node",
    # Phase 3: Plan
    "plan_edits_node",
    # Phase 4: Execute
    "route_to_edit_workers",
    "execute_structure_edits_worker",
    "execute_generation_edit_worker",
    "execute_removal_edit_worker",
    "assemble_edits_node",
    # Phase 5: Verify Structure
    "verify_structure_node",
    "check_structure_complete",
    # Phase 6: Detect Citations
    "detect_citations_node",
    "route_to_enhance_or_polish",
    # Phase 7: Enhance (when has_citations)
    "route_to_enhance_sections",
    "enhance_section_worker",
    "assemble_enhancements_node",
    "enhance_coherence_review_node",
    "route_enhance_iteration",
    # Phase 8: Verify Facts (when has_citations)
    "screen_sections_for_fact_check",
    "route_to_fact_check_sections",
    "fact_check_section_worker",
    "assemble_fact_checks_node",
    "pre_validate_citations",
    "route_to_reference_check_sections",
    "reference_check_section_worker",
    "assemble_reference_checks_node",
    "apply_verified_edits_node",
    # Phase 9: Polish
    "polish_node",
    # Phase 10: Finalize
    "finalize_node",
]
