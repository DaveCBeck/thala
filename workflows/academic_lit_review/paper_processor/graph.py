"""LangGraph definition for paper processing subgraph."""

from langgraph.graph import END, START, StateGraph

from .nodes import acquire_and_process_papers_node, extract_summaries_node
from .language_verification import (
    language_verification_node,
    should_verify_language,
)
from .types import PaperProcessingState


def create_paper_processing_subgraph() -> StateGraph:
    """Create the paper processing subgraph.

    Flow:
        START -> acquire_and_process -> [verify_language?] -> extract_summaries -> END

    The acquire_and_process node uses a unified pipeline where each paper
    goes through acquisition and processing as one unit. This naturally
    rate-limits retrieval requests since processing takes time.

    Language verification is optional - only runs for non-English language configs.
    It filters papers to ensure they actually match the target language before
    the expensive LLM extraction step.
    """
    builder = StateGraph(PaperProcessingState)

    # Add nodes
    builder.add_node("acquire_and_process", acquire_and_process_papers_node)
    builder.add_node("verify_language", language_verification_node)
    builder.add_node("extract_summaries", extract_summaries_node)

    # Add edges with conditional language verification
    builder.add_edge(START, "acquire_and_process")
    builder.add_conditional_edges(
        "acquire_and_process",
        should_verify_language,
        {
            "verify": "verify_language",
            "skip": "extract_summaries",
        },
    )
    builder.add_edge("verify_language", "extract_summaries")
    builder.add_edge("extract_summaries", END)

    return builder.compile()


paper_processing_subgraph = create_paper_processing_subgraph()
