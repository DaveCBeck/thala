"""Diffusion engine subgraph definition and assembly."""

from langgraph.graph import END, START, StateGraph

from .types import DiffusionEngineState
from .stage_nodes import (
    initialize_diffusion,
    select_expansion_seeds,
    run_citation_expansion_node,
)
from .relevance_filters import (
    enrich_with_cocitation_counts_node,
    score_relevance_node,
)
from .corpus_manager import update_corpus_and_graph
from .termination import (
    check_saturation_node,
    finalize_diffusion,
    should_continue_diffusion,
)


def create_diffusion_engine_subgraph() -> StateGraph:
    """Create the diffusion engine subgraph.

    Flow:
        START -> initialize -> select_seeds -> citation_expansion -> enrich_cocitations
              -> llm_scoring -> update_corpus -> check_saturation
              -> (continue: select_seeds OR finalize: finalize -> END)
    """
    builder = StateGraph(DiffusionEngineState)

    # Add nodes
    builder.add_node("initialize", initialize_diffusion)
    builder.add_node("select_seeds", select_expansion_seeds)
    builder.add_node("citation_expansion", run_citation_expansion_node)
    builder.add_node("enrich_cocitations", enrich_with_cocitation_counts_node)
    builder.add_node("llm_scoring", score_relevance_node)
    builder.add_node("update_corpus", update_corpus_and_graph)
    builder.add_node("check_saturation", check_saturation_node)
    builder.add_node("finalize", finalize_diffusion)

    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "select_seeds")
    builder.add_edge("select_seeds", "citation_expansion")
    builder.add_edge("citation_expansion", "enrich_cocitations")
    builder.add_edge("enrich_cocitations", "llm_scoring")
    builder.add_edge("llm_scoring", "update_corpus")
    builder.add_edge("update_corpus", "check_saturation")

    # Conditional edge: continue or finalize
    builder.add_conditional_edges(
        "check_saturation",
        should_continue_diffusion,
        {
            "continue": "select_seeds",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    return builder.compile()


diffusion_engine_subgraph = create_diffusion_engine_subgraph()
