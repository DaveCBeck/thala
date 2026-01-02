"""
Graph construction for wrapped research workflow.

Builds a LangGraph StateGraph that orchestrates web research, academic
lit review, and book finding with checkpointing between phases.
"""

from langgraph.graph import END, START, StateGraph

from workflows.wrapped.state import WrappedResearchState
from workflows.wrapped.checkpointing import save_checkpoint
from workflows.wrapped.nodes import (
    run_parallel_research,
    generate_book_query,
    run_book_finding,
    save_to_top_of_mind,
    generate_final_summary,
)


# =============================================================================
# Checkpoint-enabled node wrappers
# =============================================================================


async def parallel_research_with_checkpoint(state: WrappedResearchState) -> dict:
    """Run parallel research and save checkpoint."""
    result = await run_parallel_research(state)
    # Merge result into state for checkpointing
    updated_state = {**state, **result}
    save_checkpoint(updated_state, "parallel_research")
    return result


async def book_query_with_checkpoint(state: WrappedResearchState) -> dict:
    """Generate book query and save checkpoint."""
    result = await generate_book_query(state)
    updated_state = {**state, **result}
    save_checkpoint(updated_state, "book_query_generated")
    return result


async def book_finding_with_checkpoint(state: WrappedResearchState) -> dict:
    """Run book finding and save checkpoint."""
    result = await run_book_finding(state)
    updated_state = {**state, **result}
    save_checkpoint(updated_state, "book_finding")
    return result


async def save_to_top_of_mind_with_checkpoint(state: WrappedResearchState) -> dict:
    """Save to top_of_mind and save checkpoint."""
    result = await save_to_top_of_mind(state)
    updated_state = {**state, **result}
    save_checkpoint(updated_state, "saved_to_top_of_mind")
    return result


# =============================================================================
# Graph construction
# =============================================================================


def create_wrapped_research_graph() -> StateGraph:
    """Create the wrapped research orchestration graph.

    Flow:
        START -> parallel_research (web + academic simultaneously)
              -> generate_book_query (synthesize theme for books)
              -> book_finding (find and process books)
              -> generate_final_summary (synthesize all three)
              -> save_to_top_of_mind (save 4 records)
              -> END

    Checkpoints are saved after each major phase to enable resumption.
    """
    builder = StateGraph(WrappedResearchState)

    # Add nodes with checkpointing wrappers
    builder.add_node("parallel_research", parallel_research_with_checkpoint)
    builder.add_node("generate_book_query", book_query_with_checkpoint)
    builder.add_node("book_finding", book_finding_with_checkpoint)
    builder.add_node("generate_final_summary", generate_final_summary)
    builder.add_node("save_to_top_of_mind", save_to_top_of_mind_with_checkpoint)

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
