"""
Graph construction for wrapped research workflow.

Builds a LangGraph StateGraph that orchestrates web research, academic
lit review, and book finding.
"""

from langgraph.graph import END, START, StateGraph

from workflows.wrapped.state import WrappedResearchState
from workflows.wrapped.nodes import (
    run_parallel_research,
    generate_book_query,
    run_book_finding,
    save_to_top_of_mind,
    generate_final_summary,
)


def create_wrapped_research_graph() -> StateGraph:
    """Create the wrapped research orchestration graph.

    Flow:
        START -> parallel_research (web + academic simultaneously)
              -> generate_book_query (synthesize theme for books)
              -> book_finding (find and process books)
              -> generate_final_summary (synthesize all three)
              -> save_to_top_of_mind (save 4 records)
              -> END
    """
    builder = StateGraph(WrappedResearchState)

    # Add nodes
    builder.add_node("parallel_research", run_parallel_research)
    builder.add_node("generate_book_query", generate_book_query)
    builder.add_node("book_finding", run_book_finding)
    builder.add_node("generate_final_summary", generate_final_summary)
    builder.add_node("save_to_top_of_mind", save_to_top_of_mind)

    # Linear flow (sequential after parallel research)
    builder.add_edge(START, "parallel_research")
    builder.add_edge("parallel_research", "generate_book_query")
    builder.add_edge("generate_book_query", "book_finding")
    builder.add_edge("book_finding", "generate_final_summary")
    builder.add_edge("generate_final_summary", "save_to_top_of_mind")
    builder.add_edge("save_to_top_of_mind", END)

    return builder.compile()


# Compiled graph instance
wrapped_research_graph = create_wrapped_research_graph()
