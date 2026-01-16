"""Graph construction for synthesis workflow.

Creates a StateGraph with conditional routing based on quality settings,
parallel research workers via Send(), and sequential phases.
"""

import logging

from langgraph.graph import END, START, StateGraph

from workflows.wrappers.synthesis.state import SynthesisState
from workflows.wrappers.synthesis.nodes import (
    # Phase 1
    run_lit_review,
    # Phase 2
    run_supervision,
    # Phase 3
    generate_research_targets,
    route_to_parallel_research,
    web_research_worker,
    book_finding_worker,
    aggregate_research,
    # Phase 4
    suggest_structure,
    simple_synthesis,
    select_books,
    fetch_book_summaries,
    route_to_section_workers,
    write_section_worker,
    check_section_quality,
    assemble_sections,
    # Phase 5
    run_editing,
    # Finalize
    finalize,
)

logger = logging.getLogger(__name__)


def route_after_supervision(state: SynthesisState) -> str:
    """Route after supervision based on quality settings."""
    return "generate_research_targets"


def route_synthesis_path(state: SynthesisState) -> str:
    """Route to simple or structured synthesis based on quality settings."""
    quality_settings = state.get("quality_settings", {})

    if quality_settings.get("simple_synthesis", False):
        return "simple_synthesis"
    return "suggest_structure"


def route_after_structure(state: SynthesisState) -> str:
    """Route after structure suggestion."""
    return "select_books"


def route_after_book_summaries(state: SynthesisState) -> str:
    """Route after fetching book summaries."""
    return "section_router"


def route_after_assemble(state: SynthesisState) -> str:
    """Route after assembling sections."""
    return "run_editing"


def route_after_simple_synthesis(state: SynthesisState) -> str:
    """Route after simple synthesis."""
    return "run_editing"


def create_synthesis_graph() -> StateGraph:
    """Create the synthesis workflow graph.

    Flow:
        START
          │
          ▼
        [Phase 1: run_lit_review]
          │
          ▼
        [Phase 2: run_supervision] ←─── (may skip based on quality)
          │
          ▼
        [Phase 3: generate_research_targets]
          │
          ▼
        [Phase 3b: parallel_router] ─┬─> web_research_worker (×N) ─┬─> aggregate_research
                                     └─> book_finding_worker (×N) ─┘
          │
          ▼
        [Route: simple_synthesis vs structured]
          │
          ├── simple_synthesis ──────────────────────────────────────┐
          │                                                          │
          ▼                                                          │
        [Phase 4a: suggest_structure]                                │
          │                                                          │
          ▼                                                          │
        [Phase 4b: select_books]                                     │
          │                                                          │
          ▼                                                          │
        [Phase 4c: fetch_book_summaries]                             │
          │                                                          │
          ▼                                                          │
        [Phase 4d: section_router] ─> write_section_worker (×N) ─> assemble_sections
          │                                                          │
          ▼                                                          │
        [Phase 4e: check_section_quality] ─────────────────────────┐ │
          │                                                        │ │
          ▼                                                        │ │
        [assemble_sections] <──────────────────────────────────────┘ │
          │                                                          │
          └──────────────────────────────────────────────────────────┤
                                                                     │
          ▼                                                          │
        [Phase 5: run_editing] <─────────────────────────────────────┘
          │
          ▼
        [finalize]
          │
          ▼
        END
    """
    builder = StateGraph(SynthesisState)

    # === Add Nodes ===

    # Phase 1: Literature Review
    builder.add_node("run_lit_review", run_lit_review)

    # Phase 2: Supervision
    builder.add_node("run_supervision", run_supervision)

    # Phase 3: Research Targets
    builder.add_node("generate_research_targets", generate_research_targets)

    # Phase 3b: Parallel Research Workers
    builder.add_node("web_research_worker", web_research_worker)
    builder.add_node("book_finding_worker", book_finding_worker)
    builder.add_node("aggregate_research", aggregate_research)

    # Phase 4: Synthesis (two paths)
    builder.add_node("simple_synthesis", simple_synthesis)
    builder.add_node("suggest_structure", suggest_structure)
    builder.add_node("select_books", select_books)
    builder.add_node("fetch_book_summaries", fetch_book_summaries)
    builder.add_node("write_section_worker", write_section_worker)
    builder.add_node("check_section_quality", check_section_quality)
    builder.add_node("assemble_sections", assemble_sections)

    # Phase 5: Editing
    builder.add_node("run_editing", run_editing)

    # Finalize
    builder.add_node("finalize", finalize)

    # Router nodes (pass-through for conditional edges)
    def parallel_router(state: dict) -> dict:
        """Pass-through node for parallel dispatch."""
        return {}

    def section_router(state: dict) -> dict:
        """Pass-through node for section dispatch."""
        return {}

    def synthesis_router(state: dict) -> dict:
        """Pass-through node for synthesis path routing."""
        return {}

    builder.add_node("parallel_router", parallel_router)
    builder.add_node("section_router", section_router)
    builder.add_node("synthesis_router", synthesis_router)

    # === Add Edges ===

    # Linear: START -> lit_review -> supervision -> research_targets
    builder.add_edge(START, "run_lit_review")
    builder.add_edge("run_lit_review", "run_supervision")
    builder.add_edge("run_supervision", "generate_research_targets")

    # research_targets -> parallel_router
    builder.add_edge("generate_research_targets", "parallel_router")

    # Parallel dispatch to research workers
    builder.add_conditional_edges(
        "parallel_router",
        route_to_parallel_research,
        ["web_research_worker", "book_finding_worker"],
    )

    # Workers converge to aggregate
    builder.add_edge("web_research_worker", "aggregate_research")
    builder.add_edge("book_finding_worker", "aggregate_research")

    # After aggregation, route to synthesis path
    builder.add_edge("aggregate_research", "synthesis_router")

    # Route to simple or structured synthesis
    builder.add_conditional_edges(
        "synthesis_router",
        route_synthesis_path,
        {
            "simple_synthesis": "simple_synthesis",
            "suggest_structure": "suggest_structure",
        },
    )

    # Simple synthesis path -> editing
    builder.add_edge("simple_synthesis", "run_editing")

    # Structured synthesis path
    builder.add_edge("suggest_structure", "select_books")
    builder.add_edge("select_books", "fetch_book_summaries")
    builder.add_edge("fetch_book_summaries", "section_router")

    # Section dispatch and convergence
    builder.add_conditional_edges(
        "section_router",
        route_to_section_workers,
        ["write_section_worker", "assemble_sections"],
    )
    builder.add_edge("write_section_worker", "check_section_quality")
    builder.add_edge("check_section_quality", "assemble_sections")

    # Assemble -> editing
    builder.add_edge("assemble_sections", "run_editing")

    # Final edges
    builder.add_edge("run_editing", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create default graph instance
synthesis_graph = create_synthesis_graph()
