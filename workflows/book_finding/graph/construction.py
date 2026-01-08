"""
Graph construction for book finding workflow.

Creates a StateGraph with parallel recommendation generation,
followed by sequential search, processing, and synthesis.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from workflows.book_finding.state import BookFindingState
from workflows.book_finding.nodes import (
    generate_analogous_recommendations,
    generate_inspiring_recommendations,
    generate_expressive_recommendations,
    search_books,
    process_books,
    synthesize_output,
)


def route_to_recommendation_generators(state: BookFindingState) -> list[Send]:
    """Dispatch 3 parallel Opus calls for recommendations.

    Each generator receives the theme and optional brief,
    and generates recommendations for its category.
    """
    theme = state["input"]["theme"]
    brief = state["input"].get("brief")

    return [
        Send("generate_analogous", {"theme": theme, "brief": brief}),
        Send("generate_inspiring", {"theme": theme, "brief": brief}),
        Send("generate_expressive", {"theme": theme, "brief": brief}),
    ]


def create_book_finding_graph() -> StateGraph:
    """Create the book finding workflow graph.

    Flow:
        START -> [3 parallel recommendation generators] -> search_books
              -> process_books -> synthesize_output -> END

    The three recommendation generators run in parallel using Send(),
    then their outputs are aggregated via the add reducer on the
    *_recommendations fields in BookFindingState.
    """
    builder = StateGraph(BookFindingState)

    # Add nodes
    builder.add_node("generate_analogous", generate_analogous_recommendations)
    builder.add_node("generate_inspiring", generate_inspiring_recommendations)
    builder.add_node("generate_expressive", generate_expressive_recommendations)
    builder.add_node("search_books", search_books)
    builder.add_node("process_books", process_books)
    builder.add_node("synthesize", synthesize_output)

    # Parallel dispatch to recommendation generators
    builder.add_conditional_edges(
        START,
        route_to_recommendation_generators,
        ["generate_analogous", "generate_inspiring", "generate_expressive"],
    )

    # All generators converge to search
    builder.add_edge("generate_analogous", "search_books")
    builder.add_edge("generate_inspiring", "search_books")
    builder.add_edge("generate_expressive", "search_books")

    # Linear flow after convergence
    builder.add_edge("search_books", "process_books")
    builder.add_edge("process_books", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


book_finding_graph = create_book_finding_graph()
