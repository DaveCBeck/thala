"""LangGraph construction for substack_review workflow.

Transforms academic literature reviews into Substack-style essays through:
1. Input validation
2. Parallel generation of 3 essay variants
3. Structured selection of the best essay
4. Reference formatting via Zotero
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .state import SubstackReviewState
from .nodes import (
    validate_input_node,
    write_puzzle_essay,
    write_finding_essay,
    write_contrarian_essay,
    choose_essay_node,
    format_references_node,
)


def route_after_validation(state: SubstackReviewState) -> list[Send] | str:
    """Route to parallel writing agents or end on validation failure."""
    if not state.get("is_valid"):
        return END

    # Fan out to 3 parallel writing agents
    lit_review = state["input"]["literature_review"]
    return [
        Send("write_puzzle", {"literature_review": lit_review, "input": state["input"]}),
        Send("write_finding", {"literature_review": lit_review, "input": state["input"]}),
        Send("write_contrarian", {"literature_review": lit_review, "input": state["input"]}),
    ]


def create_substack_review_graph() -> StateGraph:
    """Create the workflow graph.

    Flow:
        START -> validate_input
              -> [3 parallel writing agents via Send()]
              -> choose_essay (structured output)
              -> format_references (Zotero lookup)
              -> END

    The three writing agents run in parallel using Send(),
    then their outputs are aggregated via the add reducer on
    essay_drafts in SubstackReviewState.
    """
    builder = StateGraph(SubstackReviewState)

    # Add nodes
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("write_puzzle", write_puzzle_essay)
    builder.add_node("write_finding", write_finding_essay)
    builder.add_node("write_contrarian", write_contrarian_essay)
    builder.add_node("choose_essay", choose_essay_node)
    builder.add_node("format_references", format_references_node)

    # Entry point
    builder.add_edge(START, "validate_input")

    # Conditional fan-out after validation
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        ["write_puzzle", "write_finding", "write_contrarian", END],
    )

    # All writers converge to chooser
    builder.add_edge("write_puzzle", "choose_essay")
    builder.add_edge("write_finding", "choose_essay")
    builder.add_edge("write_contrarian", "choose_essay")

    # Linear flow after selection
    builder.add_edge("choose_essay", "format_references")
    builder.add_edge("format_references", END)

    return builder.compile()


# Export compiled graph
substack_review_graph = create_substack_review_graph()
