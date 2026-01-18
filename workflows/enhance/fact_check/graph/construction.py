"""Graph construction for the fact-check workflow."""

import logging

from langgraph.graph import StateGraph, START, END

from workflows.enhance.fact_check.state import FactCheckState
from workflows.enhance.fact_check.nodes import (
    # Phase 1: Parse
    parse_document_node,
    # Phase 2: Detect Citations
    detect_citations_node,
    route_citations_or_finalize,
    # Phase 3: Screen
    screen_sections_for_fact_check,
    # Phase 4: Fact-check
    route_to_fact_check_sections,
    fact_check_section_worker,
    assemble_fact_checks_node,
    # Phase 5: Reference-check
    pre_validate_citations,
    route_to_reference_check_sections,
    reference_check_section_worker,
    assemble_reference_checks_node,
    # Phase 6: Apply Edits
    apply_verified_edits_node,
    # Phase 7: Finalize
    finalize_node,
)

logger = logging.getLogger(__name__)


def create_fact_check_graph() -> StateGraph:
    """Create the fact-check workflow graph.

    Graph structure:
    1. Parse document (or use provided model)
    2. Detect citations
    3. If no citations -> finalize (skip)
    4. Screen sections for fact-check priority
    5. Fact-check sections (parallel)
    6. Pre-validate citations
    7. Reference-check sections (parallel)
    8. Apply verified edits
    9. Finalize and output

    Returns:
        Compiled StateGraph
    """
    builder = StateGraph(FactCheckState)

    # === Add Nodes ===

    # Phase 1: Parse
    builder.add_node("parse_document", parse_document_node)

    # Phase 2: Detect Citations
    builder.add_node("detect_citations", detect_citations_node)

    # Phase 3: Screen (via router)
    def fact_check_router_node(state: dict) -> dict:
        """Pass-through node for fact-check routing."""
        return {}

    builder.add_node("fact_check_router", fact_check_router_node)
    builder.add_node("screen_fact_check", screen_sections_for_fact_check)

    # Phase 4: Fact-check
    builder.add_node("fact_check_section", fact_check_section_worker)
    builder.add_node("assemble_fact_checks", assemble_fact_checks_node)

    # Phase 5: Reference-check
    builder.add_node("pre_validate_citations", pre_validate_citations)

    def reference_check_router_node(state: dict) -> dict:
        """Pass-through node for reference-check routing."""
        return {}

    builder.add_node("reference_check_router", reference_check_router_node)
    builder.add_node("reference_check_section", reference_check_section_worker)
    builder.add_node("assemble_reference_checks", assemble_reference_checks_node)

    # Phase 6: Apply Edits
    builder.add_node("apply_verified_edits", apply_verified_edits_node)

    # Phase 7: Finalize
    builder.add_node("finalize", finalize_node)

    # === Add Edges ===

    # Linear flow: start -> parse -> detect_citations
    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "detect_citations")

    # Conditional routing based on citations
    builder.add_conditional_edges(
        "detect_citations",
        route_citations_or_finalize,
        {
            "fact_check_router": "fact_check_router",  # Has citations
            "finalize": "finalize",  # No citations -> skip
        },
    )

    # Fact-check routing: router -> screening -> workers
    builder.add_edge("fact_check_router", "screen_fact_check")

    # After screening, route to fact-check workers (parallel via Send)
    builder.add_conditional_edges(
        "screen_fact_check",
        route_to_fact_check_sections,
        [
            "fact_check_section",
            "reference_check_router",  # When no sections to check
        ],
    )

    # Fact-check workers -> assemble -> pre-validate citations -> reference router
    builder.add_edge("fact_check_section", "assemble_fact_checks")
    builder.add_edge("assemble_fact_checks", "pre_validate_citations")
    builder.add_edge("pre_validate_citations", "reference_check_router")

    # Route to reference-check workers (parallel via Send)
    builder.add_conditional_edges(
        "reference_check_router",
        route_to_reference_check_sections,
        [
            "reference_check_section",
            "apply_verified_edits",  # When no sections to check
        ],
    )

    # Reference-check workers -> assemble -> apply edits
    builder.add_edge("reference_check_section", "assemble_reference_checks")
    builder.add_edge("assemble_reference_checks", "apply_verified_edits")

    # Apply verified edits -> finalize -> end
    builder.add_edge("apply_verified_edits", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create default graph instance
fact_check_graph = create_fact_check_graph()
