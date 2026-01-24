"""Graph construction for the editing workflow.

Uses V2 structure phase (analyze → rewrite → reassemble) followed by
V1 Enhancement and Polish phases via bridge node.
"""

import logging

from langgraph.graph import StateGraph, START, END

from workflows.enhance.editing.state import EditingState
from workflows.enhance.editing.nodes import (
    # V2 Structure Phase
    v2_analyze_node,
    v2_rewrite_router_node,
    v2_route_to_rewriters,
    v2_rewrite_section_node,
    v2_reassemble_node,
    # Bridge (V2 -> V1)
    v2_to_v1_bridge_node,
    # Citation routing
    route_to_enhance_or_polish,
    # V1 Enhancement Phase
    route_to_enhance_sections,
    enhance_section_worker,
    assemble_enhancements_node,
    enhance_coherence_review_node,
    route_enhance_iteration,
    # V1 Polish Phase
    polish_node,
    # V1 Finalize
    finalize_node,
)

logger = logging.getLogger(__name__)


def create_editing_graph() -> StateGraph:
    """Create the editing workflow graph.

    Graph structure:
    1. V2 Structure Phase:
       - analyze: Identify sections needing work
       - rewrite_router -> rewrite_section (parallel): Rewrite flagged sections
       - reassemble: Combine results and verify coherence

    2. Bridge: Convert V2 markdown output to V1 DocumentModel

    3. Citation-based routing:
       - If citations: Enhancement phase
       - If no citations: Skip to Polish

    4. V1 Enhancement Phase (when has_citations):
       - enhance_router -> enhance_section (parallel)
       - assemble_enhancements -> enhance_coherence_review
       - Loop if more iterations needed

    5. V1 Polish Phase:
       - polish: Final language and flow improvements

    6. Finalize:
       - finalize: Package output

    Returns:
        Compiled StateGraph
    """
    builder = StateGraph(EditingState)

    # === V2 Structure Phase Nodes ===
    builder.add_node("analyze", v2_analyze_node)
    builder.add_node("rewrite_router", v2_rewrite_router_node)
    builder.add_node("rewrite_section", v2_rewrite_section_node)
    builder.add_node("reassemble", v2_reassemble_node)

    # === Bridge Node ===
    builder.add_node("bridge", v2_to_v1_bridge_node)

    # === V1 Enhancement Phase Nodes ===
    def enhance_router_node(state: dict) -> dict:
        """Pass-through node for enhancement routing."""
        return {}

    builder.add_node("enhance_router", enhance_router_node)
    builder.add_node("enhance_section", enhance_section_worker)
    builder.add_node("assemble_enhancements", assemble_enhancements_node)
    builder.add_node("enhance_coherence_review", enhance_coherence_review_node)

    # === V1 Polish Phase Nodes ===
    builder.add_node("polish", polish_node)

    # === Finalize Node ===
    builder.add_node("finalize", finalize_node)

    # === V2 Structure Phase Edges ===
    # START -> analyze -> rewrite_router
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "rewrite_router")

    # rewrite_router conditionally routes to rewrite_section workers or reassemble
    builder.add_conditional_edges(
        "rewrite_router",
        v2_route_to_rewriters,
        ["rewrite_section", "reassemble"],
    )

    # rewrite_section workers -> reassemble
    builder.add_edge("rewrite_section", "reassemble")

    # reassemble -> bridge
    builder.add_edge("reassemble", "bridge")

    # === Bridge -> Citation Routing ===
    builder.add_conditional_edges(
        "bridge",
        route_to_enhance_or_polish,
        {
            "enhance": "enhance_router",
            "polish": "polish",
        },
    )

    # === V1 Enhancement Phase Edges ===
    # enhance_router -> enhance_section workers (parallel via Send)
    builder.add_conditional_edges(
        "enhance_router",
        route_to_enhance_sections,
        [
            "enhance_section",
            "enhance_coherence_review",  # When no sections to enhance
        ],
    )

    # enhance_section workers -> assemble_enhancements -> enhance_coherence_review
    builder.add_edge("enhance_section", "assemble_enhancements")
    builder.add_edge("assemble_enhancements", "enhance_coherence_review")

    # enhance_coherence_review routes to continue or polish
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
            "enhance_router": "enhance_router",
            "polish": "polish",
        },
    )

    # === V1 Polish and Finalize Edges ===
    builder.add_edge("polish", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create default graph instance
editing_graph = create_editing_graph()
