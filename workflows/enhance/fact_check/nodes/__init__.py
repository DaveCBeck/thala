"""Node implementations for the fact-check workflow."""

from .parse_document import parse_document_node
from .detect_citations import (
    detect_citations_node,
    route_citations_or_finalize,
)
from .screen_sections import screen_sections_for_fact_check
from .fact_check import (
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
from .apply_edits import apply_verified_edits_node
from .finalize import finalize_node

__all__ = [
    # Phase 1: Parse
    "parse_document_node",
    # Phase 2: Detect Citations
    "detect_citations_node",
    "route_citations_or_finalize",
    # Phase 3: Screen
    "screen_sections_for_fact_check",
    # Phase 4: Fact-check
    "route_to_fact_check_sections",
    "fact_check_section_worker",
    "assemble_fact_checks_node",
    # Phase 5: Reference-check
    "pre_validate_citations",
    "route_to_reference_check_sections",
    "reference_check_section_worker",
    "assemble_reference_checks_node",
    # Phase 6: Apply Edits
    "apply_verified_edits_node",
    # Phase 7: Finalize
    "finalize_node",
]
