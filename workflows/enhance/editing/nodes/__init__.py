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
    # Phase 8: Polish
    "polish_node",
    # Phase 9: Finalize
    "finalize_node",
]
