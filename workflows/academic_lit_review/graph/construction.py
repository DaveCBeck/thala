"""Graph construction for academic literature review workflow.

This graph produces a literature review WITHOUT supervision loops.
For supervised review with quality loops, use workflows.supervised_lit_review.
"""

from langgraph.graph import END, START, StateGraph

from workflows.academic_lit_review.state import (
    AcademicLitReviewState,
)
from .phases import (
    discovery_phase_node,
    diffusion_phase_node,
    processing_phase_node,
    clustering_phase_node,
    synthesis_phase_node,
)


def create_academic_lit_review_graph() -> StateGraph:
    """Create the main academic literature review workflow graph.

    Flow:
        START -> discovery -> diffusion -> processing
              -> clustering -> synthesis -> END

    Note: Supervision loops are NOT included in this graph.
    Use supervised_lit_review for reviews with quality supervision.
    """
    builder = StateGraph(AcademicLitReviewState)

    # Add phase nodes
    builder.add_node("discovery", discovery_phase_node)
    builder.add_node("diffusion", diffusion_phase_node)
    builder.add_node("processing", processing_phase_node)
    builder.add_node("clustering", clustering_phase_node)
    builder.add_node("synthesis", synthesis_phase_node)

    # Add edges (linear flow)
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "diffusion")
    builder.add_edge("diffusion", "processing")
    builder.add_edge("processing", "clustering")
    builder.add_edge("clustering", "synthesis")
    builder.add_edge("synthesis", END)

    return builder.compile()


# Export compiled graph
academic_lit_review_graph = create_academic_lit_review_graph()
