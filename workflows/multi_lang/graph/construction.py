"""
Graph construction for multi_lang research workflow.

Builds a LangGraph StateGraph that orchestrates research across multiple
languages with relevance checking, sequential execution, and synthesis.
"""

from langgraph.graph import END, START, StateGraph

from workflows.multi_lang.state import MultiLangState
from workflows.multi_lang.nodes import (
    select_languages,
    check_relevance_batch,
    filter_relevant_languages,
    execute_next_language,
    check_languages_complete,
    run_sonnet_analysis,
    run_opus_integration,
    save_multi_lang_results,
)
from .routing import route_after_language_selection, route_language_loop


def create_multi_lang_graph() -> StateGraph:
    """
    Create the multi_lang workflow graph.

    Flow for Mode 1 (Set Languages):
        START -> select_languages -> execute_next_language (loop)
              -> sonnet_analysis -> opus_integration -> save_results -> END

    Flow for Mode 2 (All Languages):
        START -> select_languages -> check_relevance_batch
              -> filter_relevant_languages -> execute_next_language (loop)
              -> sonnet_analysis -> opus_integration -> save_results -> END
    """
    builder = StateGraph(MultiLangState)

    # Phase 1: Language Selection
    builder.add_node("select_languages", select_languages)

    # Phase 2: Relevance Checking (Mode 2 only)
    builder.add_node("check_relevance_batch", check_relevance_batch)
    builder.add_node("filter_relevant_languages", filter_relevant_languages)

    # Phase 3: Sequential Language Execution
    builder.add_node("execute_next_language", execute_next_language)
    builder.add_node("check_languages_complete", check_languages_complete)

    # Phase 4: Cross-Language Synthesis
    builder.add_node("sonnet_analysis", run_sonnet_analysis)
    builder.add_node("opus_integration", run_opus_integration)

    # Phase 5: Save Results
    builder.add_node("save_results", save_multi_lang_results)

    # Edges
    builder.add_edge(START, "select_languages")

    # Conditional: mode determines path after selection
    builder.add_conditional_edges(
        "select_languages",
        route_after_language_selection,
        ["execute_next_language", "check_relevance_batch"],
    )

    # Relevance checking path (mode 2)
    builder.add_edge("check_relevance_batch", "filter_relevant_languages")
    builder.add_edge("filter_relevant_languages", "execute_next_language")

    # Language execution loop
    builder.add_edge("execute_next_language", "check_languages_complete")
    builder.add_conditional_edges(
        "check_languages_complete",
        route_language_loop,
        ["execute_next_language", "sonnet_analysis"],
    )

    # Synthesis phase
    builder.add_edge("sonnet_analysis", "opus_integration")
    builder.add_edge("opus_integration", "save_results")
    builder.add_edge("save_results", END)

    return builder.compile()


# Compiled graph instance
multi_lang_graph = create_multi_lang_graph()
