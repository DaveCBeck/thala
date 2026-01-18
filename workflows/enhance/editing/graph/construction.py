"""Graph construction for the editing workflow."""

import logging

from langgraph.graph import StateGraph, START, END

from workflows.enhance.editing.state import EditingState
from workflows.enhance.editing.nodes import (
    # Phase 1: Parse
    parse_document_node,
    # Phase 2: Analyze
    analyze_structure_node,
    # Phase 3: Plan
    plan_edits_node,
    # Phase 4: Execute
    route_to_edit_workers,
    execute_structure_edits_worker,
    execute_generation_edit_worker,
    execute_removal_edit_worker,
    assemble_edits_node,
    # Phase 5: Verify Structure
    verify_structure_node,
    check_structure_complete,
    # Phase 6: Detect Citations
    detect_citations_node,
    route_to_enhance_or_polish,
    # Phase 7: Enhance (when has_citations)
    route_to_enhance_sections,
    enhance_section_worker,
    assemble_enhancements_node,
    enhance_coherence_review_node,
    route_enhance_iteration,
    # Phase 8: Polish
    polish_node,
    # Phase 9: Finalize
    finalize_node,
)

logger = logging.getLogger(__name__)


def create_editing_graph() -> StateGraph:
    """Create the editing workflow graph.

    Graph structure:
    1. Parse document into structured model
    2. Analyze structure to identify issues
    3. Plan edits based on issues
    4. Execute edits (parallel workers for generation, sequential for structure)
    5. Assemble edits and verify structure
    6. Loop back if more structure work needed
    7. Detect citations in document
    8. If citations found: Enhance sections (with iteration loop)
    9. Polish for flow and coherence
    10. Finalize and output

    Note: Fact-check and reference-check are now in a separate workflow
    (workflows.enhance.fact_check) that runs after editing.

    Returns:
        Compiled StateGraph
    """
    builder = StateGraph(EditingState)

    # === Add Nodes ===

    # Phase 1: Parse
    builder.add_node("parse_document", parse_document_node)

    # Phase 2: Analyze
    builder.add_node("analyze_structure", analyze_structure_node)

    # Phase 3: Plan
    builder.add_node("plan_edits", plan_edits_node)

    # Phase 4: Execute (workers)
    builder.add_node("execute_structure_edits", execute_structure_edits_worker)
    builder.add_node("execute_generation_edit", execute_generation_edit_worker)
    builder.add_node("execute_removal_edit", execute_removal_edit_worker)
    builder.add_node("assemble_edits", assemble_edits_node)

    # Phase 5: Verify Structure
    builder.add_node("verify_structure", verify_structure_node)

    # Phase 6: Detect Citations
    builder.add_node("detect_citations", detect_citations_node)

    # Phase 7: Enhance (when has_citations)
    builder.add_node("enhance_section", enhance_section_worker)
    builder.add_node("assemble_enhancements", assemble_enhancements_node)
    builder.add_node("enhance_coherence_review", enhance_coherence_review_node)

    # Phase 8: Polish
    builder.add_node("polish", polish_node)

    # Phase 9: Finalize
    builder.add_node("finalize", finalize_node)

    # === Add Edges ===

    # Linear flow: start -> parse -> analyze -> plan
    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "analyze_structure")
    builder.add_edge("analyze_structure", "plan_edits")

    # Conditional routing to edit workers
    builder.add_conditional_edges(
        "plan_edits",
        route_to_edit_workers,
        [
            "execute_structure_edits",
            "execute_generation_edit",
            "execute_removal_edit",
            "assemble_edits",
        ],
    )

    # Workers -> assemble
    builder.add_edge("execute_structure_edits", "assemble_edits")
    builder.add_edge("execute_generation_edit", "assemble_edits")
    builder.add_edge("execute_removal_edit", "assemble_edits")

    # Assemble -> verify
    builder.add_edge("assemble_edits", "verify_structure")

    # Structure iteration loop - routes to detect_citations when done
    builder.add_conditional_edges(
        "verify_structure",
        check_structure_complete,
        {
            "continue_structure": "analyze_structure",  # Loop back
            "proceed_to_polish": "detect_citations",  # Go to citation detection
        },
    )

    # === Citation Detection & Routing ===
    # Detect if document has citations -> route to enhance or polish
    builder.add_conditional_edges(
        "detect_citations",
        route_to_enhance_or_polish,
        {
            "enhance": "enhance_router",  # Has citations -> enhance phase
            "polish": "polish",  # No citations -> skip to polish
        },
    )

    # === Enhancement Phase (when has_citations) ===
    # Add an empty router node that just passes through state
    # This is needed because we can't route directly from detect_citations
    # to the Send-based routing
    def enhance_router_node(state: dict) -> dict:
        """Pass-through node for enhancement routing."""
        return {}

    builder.add_node("enhance_router", enhance_router_node)

    # Route from enhance_router to section workers (parallel via Send)
    builder.add_conditional_edges(
        "enhance_router",
        route_to_enhance_sections,
        [
            "enhance_section",
            "enhance_coherence_review",  # When no sections to enhance
        ],
    )

    # Enhance section workers -> assemble -> coherence review
    builder.add_edge("enhance_section", "assemble_enhancements")
    builder.add_edge("assemble_enhancements", "enhance_coherence_review")

    # Coherence review routes to continue enhancing or proceed to polish
    # (Fact-check is now a separate workflow that runs after editing)
    def route_enhance_to_polish(state: dict) -> str:
        """Route to continue enhancing or proceed to polish."""
        result = route_enhance_iteration(state)
        if result == "continue":
            return "enhance_router"
        return "polish"

    builder.add_conditional_edges(
        "enhance_coherence_review",
        route_enhance_to_polish,
        {
            "enhance_router": "enhance_router",  # More iterations needed
            "polish": "polish",  # Enhancement complete -> polish
        },
    )

    # === Final Phases ===
    # Polish -> finalize -> end
    builder.add_edge("polish", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create default graph instance
editing_graph = create_editing_graph()
