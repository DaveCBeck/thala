"""LangGraph definition for paper processing subgraph."""

from langgraph.graph import END, START, StateGraph

from .nodes import acquire_and_process_papers_node, extract_summaries_node
from .types import PaperProcessingState


def create_paper_processing_subgraph() -> StateGraph:
    """Create the paper processing subgraph.

    Flow:
        START -> acquire_and_process -> extract_summaries -> END

    The acquire_and_process node uses a unified pipeline where each paper
    goes through acquisition and processing as one unit. This naturally
    rate-limits retrieval requests since processing takes time.
    """
    builder = StateGraph(PaperProcessingState)

    builder.add_node("acquire_and_process", acquire_and_process_papers_node)
    builder.add_node("extract_summaries", extract_summaries_node)

    builder.add_edge(START, "acquire_and_process")
    builder.add_edge("acquire_and_process", "extract_summaries")
    builder.add_edge("extract_summaries", END)

    return builder.compile()


paper_processing_subgraph = create_paper_processing_subgraph()
