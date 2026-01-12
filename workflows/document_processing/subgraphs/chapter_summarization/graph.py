"""LangGraph construction for chapter summarization."""

from langgraph.graph import StateGraph, START, END

from workflows.document_processing.state import DocumentProcessingState

from .nodes import summarize_chapters, aggregate_summaries


def create_chapter_summarization_subgraph():
    """
    Create subgraph for parallel chapter summarization.

    Uses single node with asyncio.gather() for true concurrent execution:
    1. START -> summarize_chapters (batches all chapters concurrently)
    2. aggregate_summaries -> END
    """
    graph = StateGraph(DocumentProcessingState)

    # Add nodes
    graph.add_node("summarize_chapters", summarize_chapters)
    graph.add_node("aggregate_summaries", aggregate_summaries)

    # Add edges
    graph.add_edge(START, "summarize_chapters")
    graph.add_edge("summarize_chapters", "aggregate_summaries")
    graph.add_edge("aggregate_summaries", END)

    return graph.compile()


# Export compiled subgraph
chapter_summarization_subgraph = create_chapter_summarization_subgraph()
