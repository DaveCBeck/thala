"""Deep research graph construction."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from workflows.research.state import DeepResearchState
from workflows.research.nodes.clarify_intent import clarify_intent
from workflows.research.nodes.create_brief import create_brief
from workflows.research.nodes.search_memory import search_memory_node
from workflows.research.nodes.iterate_plan import iterate_plan
from workflows.research.nodes.supervisor import supervisor
from workflows.research.nodes.refine_draft import refine_draft
from workflows.research.nodes.final_report import final_report
from workflows.research.nodes.process_citations import process_citations
from workflows.research.nodes.save_findings import save_findings
from workflows.research.nodes.translate_report import translate_report
from workflows.research.subgraphs.web_researcher import web_researcher_subgraph
from workflows.research.subgraphs.academic_researcher import academic_researcher_subgraph
from workflows.research.subgraphs.book_researcher import book_researcher_subgraph

from .routing import route_after_clarify, route_after_create_brief, route_supervisor_action
from .aggregation import aggregate_researcher_findings


def create_deep_research_graph():
    """
    Create the main deep research workflow graph.

    Flow:
    START -> clarify_intent -> create_brief -> search_memory -> iterate_plan
          -> supervisor <-> researcher (loop)
          -> final_report -> process_citations -> translate_report -> save_findings -> END
    """
    builder = StateGraph(DeepResearchState)

    # Add nodes
    builder.add_node("clarify_intent", clarify_intent)
    builder.add_node("create_brief", create_brief)
    builder.add_node("search_memory", search_memory_node)
    builder.add_node("iterate_plan", iterate_plan)
    builder.add_node(
        "supervisor",
        supervisor,
        retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
    )
    # Specialized researcher subgraphs
    builder.add_node("web_researcher", web_researcher_subgraph)
    builder.add_node("academic_researcher", academic_researcher_subgraph)
    builder.add_node("book_researcher", book_researcher_subgraph)
    builder.add_node("aggregate_findings", aggregate_researcher_findings)
    builder.add_node("refine_draft", refine_draft)
    builder.add_node(
        "final_report",
        final_report,
        retry=RetryPolicy(max_attempts=2, backoff_factor=2.0),
    )
    builder.add_node(
        "process_citations",
        process_citations,
        retry=RetryPolicy(max_attempts=2, backoff_factor=2.0),
    )
    builder.add_node("translate_report", translate_report)
    builder.add_node("save_findings", save_findings)

    # Entry flow
    builder.add_edge(START, "clarify_intent")
    builder.add_conditional_edges("clarify_intent", route_after_clarify, ["create_brief"])
    builder.add_edge("create_brief", "search_memory")
    builder.add_edge("search_memory", "iterate_plan")
    builder.add_edge("iterate_plan", "supervisor")

    # Supervisor routing (diffusion loop)
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_action,
        ["web_researcher", "academic_researcher", "book_researcher",
         "refine_draft", "final_report", "supervisor"],
    )

    # All researchers converge to aggregation
    builder.add_edge("web_researcher", "aggregate_findings")
    builder.add_edge("academic_researcher", "aggregate_findings")
    builder.add_edge("book_researcher", "aggregate_findings")
    builder.add_edge("aggregate_findings", "supervisor")

    # Refine draft loops back to supervisor
    builder.add_edge("refine_draft", "supervisor")

    # Final stages
    builder.add_edge("final_report", "process_citations")
    builder.add_edge("process_citations", "translate_report")
    builder.add_edge("translate_report", "save_findings")
    builder.add_edge("save_findings", END)

    return builder.compile()


deep_research_graph = create_deep_research_graph()
