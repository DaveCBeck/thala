"""
Main deep research workflow graph.

Implements the Self-Balancing Diffusion Algorithm with:
- Memory-first research (search Thala stores before web)
- Plan customization based on existing knowledge
- Parallel researcher agents (up to 3 concurrent)
- Iterative refinement with completeness checking
- Final report with citations saved to store

Flow:
1. clarify_intent - Ask clarifying questions if needed
2. create_brief - Generate structured research brief
3. search_memory - Search Thala stores for existing knowledge
4. iterate_plan - Customize plan based on memory (OPUS)
5. supervisor - Coordinate research with diffusion algorithm (OPUS)
   - Loop: generate questions -> researchers -> refine draft
6. final_report - Generate comprehensive report (OPUS)
7. save_findings - Save to store
"""

from core.config import configure_langsmith

configure_langsmith()

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from workflows.research.state import DeepResearchState, ResearcherState, DiffusionState, calculate_completeness
from workflows.research.nodes.clarify_intent import clarify_intent
from workflows.research.nodes.create_brief import create_brief
from workflows.research.nodes.search_memory import search_memory_node
from workflows.research.nodes.iterate_plan import iterate_plan
from workflows.research.nodes.supervisor import supervisor
from workflows.research.nodes.refine_draft import refine_draft
from workflows.research.nodes.final_report import final_report
from workflows.research.nodes.process_citations import process_citations
from workflows.research.nodes.save_findings import save_findings
from workflows.research.subgraphs.researcher import researcher_subgraph

logger = logging.getLogger(__name__)

# Maximum concurrent researcher agents
MAX_CONCURRENT_RESEARCHERS = 3

# Recursion limits by depth (generous buffer for complex graph paths)
RECURSION_LIMITS = {
    "quick": 50,
    "standard": 100,
    "comprehensive": 200,
}


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_clarify(state: DeepResearchState) -> str:
    """Route based on whether clarification is needed."""
    if state.get("clarification_needed"):
        # If clarification needed but no responses yet, proceed anyway
        # (In a real implementation, this would pause for user input)
        if not state.get("clarification_responses"):
            logger.info("Clarification needed but proceeding without responses")
    return "create_brief"


def route_supervisor_action(state: DeepResearchState) -> str | list[Send]:
    """Route based on supervisor's chosen action."""
    current_status = state.get("current_status", "")

    if current_status == "conduct_research":
        # Fan out to researcher agents (max 3 parallel)
        pending = state.get("pending_questions", [])[:MAX_CONCURRENT_RESEARCHERS]

        if not pending:
            logger.warning("No pending questions for research - completing")
            return "final_report"

        logger.info(f"Launching {len(pending)} researcher agents")

        return [
            Send("researcher", ResearcherState(
                question=q,
                search_queries=[],
                search_results=[],
                scraped_content=[],
                thinking=None,
                finding=None,
            ))
            for q in pending
        ]

    elif current_status == "refine_draft":
        return "refine_draft"

    elif current_status == "research_complete":
        return "final_report"

    else:
        # Default: continue to supervisor
        return "supervisor"


def aggregate_researcher_findings(state: DeepResearchState) -> dict[str, Any]:
    """Aggregate findings from researcher agents back to main state.

    This is called after researcher agents complete. The findings are
    automatically aggregated via the Annotated[..., add] pattern.
    Also updates completeness based on accumulated findings.
    """
    findings = state.get("research_findings", [])
    diffusion = state.get("diffusion", {})
    brief = state.get("research_brief", {})
    draft = state.get("draft_report")

    # Calculate updated completeness based on new findings
    new_completeness = calculate_completeness(
        findings=findings,
        key_questions=brief.get("key_questions", []),
        iteration=diffusion.get("iteration", 0),
        max_iterations=diffusion.get("max_iterations", 4),
        gaps_remaining=draft.get("gaps_remaining", []) if draft else [],
    )

    logger.info(
        f"Aggregated {len(findings)} research findings, completeness: {new_completeness:.0%}"
    )

    # Clear pending questions and update completeness
    return {
        "pending_questions": [],
        "current_status": "supervising",
        "diffusion": {
            **diffusion,
            "completeness_score": new_completeness,
        },
    }


# =============================================================================
# Graph Construction
# =============================================================================


def create_deep_research_graph():
    """
    Create the main deep research workflow graph.

    Flow:
    START -> clarify_intent -> create_brief -> search_memory -> iterate_plan
          -> supervisor <-> researcher (loop) -> final_report -> save_findings -> END
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
    builder.add_node("researcher", researcher_subgraph)
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
        ["researcher", "refine_draft", "final_report", "supervisor"],
    )

    # Researchers converge to aggregation
    builder.add_edge("researcher", "aggregate_findings")
    builder.add_edge("aggregate_findings", "supervisor")

    # Refine draft loops back to supervisor
    builder.add_edge("refine_draft", "supervisor")

    # Final stages
    builder.add_edge("final_report", "process_citations")
    builder.add_edge("process_citations", "save_findings")
    builder.add_edge("save_findings", END)

    return builder.compile()


# Export compiled graph
deep_research_graph = create_deep_research_graph()


# =============================================================================
# Public API
# =============================================================================


async def deep_research(
    query: str,
    depth: Literal["quick", "standard", "comprehensive"] = "standard",
    max_sources: int = 20,
    max_iterations: int = None,
    clarification_responses: dict[str, str] = None,
) -> DeepResearchState:
    """
    Run deep research on a topic.

    This is the main entry point for the research workflow.

    Args:
        query: Research question or topic
        depth: Research depth
            - "quick": 2 iterations, ~5 min, focused questions
            - "standard": 4 iterations, ~15 min, balanced
            - "comprehensive": 8 iterations, 30+ min, exhaustive
        max_sources: Maximum web sources to consult
        max_iterations: Override default iteration count for depth
        clarification_responses: Pre-provided clarification answers

    Returns:
        DeepResearchState with:
        - final_report: Complete research report
        - citations: List of sources used
        - store_record_id: UUID if saved to store
        - research_findings: All findings from researchers
        - errors: Any errors encountered

    Example:
        result = await deep_research(
            "Impact of AI on jobs in 2025",
            depth="standard",
        )
        print(result["final_report"])
    """
    # Generate a run_id for LangSmith tracing (allows inspection of runs)
    run_id = uuid.uuid4()

    initial_state: DeepResearchState = {
        "input": {
            "query": query,
            "depth": depth,
            "max_sources": max_sources,
            "max_iterations": max_iterations,
        },
        "clarification_needed": False,
        "clarification_questions": [],
        "clarification_responses": clarification_responses,
        "research_brief": None,
        "memory_findings": [],
        "memory_context": "",
        "research_plan": None,
        "pending_questions": [],
        "active_researchers": 0,
        "research_findings": [],
        "supervisor_messages": [],
        "diffusion": DiffusionState(
            iteration=0,
            max_iterations=max_iterations or {"quick": 2, "standard": 4, "comprehensive": 8}[depth],
            completeness_score=0.0,
            areas_explored=[],
            areas_to_explore=[],
        ),
        "draft_report": None,
        "final_report": None,
        "citations": [],
        "citation_keys": [],
        "store_record_id": None,
        "zotero_key": None,
        "errors": [],
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "current_status": "starting",
        "langsmith_run_id": str(run_id),
    }

    recursion_limit = RECURSION_LIMITS.get(depth, 100)
    logger.info(f"Starting deep research: query='{query[:50]}...', depth={depth}, recursion_limit={recursion_limit}, run_id={run_id}")

    result = await deep_research_graph.ainvoke(
        initial_state,
        config={
            "recursion_limit": recursion_limit,
            "run_id": run_id,
            "run_name": f"deep_research:{query[:30]}",
        },
    )

    logger.info(
        f"Deep research complete: status={result.get('current_status')}, "
        f"findings={len(result.get('research_findings', []))}, "
        f"iterations={result.get('diffusion', {}).get('iteration', 0)}"
    )

    return result
