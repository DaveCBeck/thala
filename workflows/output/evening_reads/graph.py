"""LangGraph construction for evening_reads workflow.

Transforms academic literature reviews into a 4-part series:
1 overview + 3 deep-dives through:
1. Input validation and citation mapping
2. Content planning with structured output
3. Parallel content fetching (3x)
4. Parallel deep-dive writing (3x)
5. Overview writing
6. Reference formatting
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .state import EveningReadsState
from .nodes import (
    validate_input_node,
    plan_content_node,
    fetch_content_node,
    write_deep_dive_node,
    write_overview_node,
    generate_images_node,
    format_references_node,
)


def route_after_validation(state: EveningReadsState) -> str:
    """Route to planning or end on validation failure."""
    if not state.get("is_valid"):
        return END
    return "plan_content"


def route_to_fetch(state: EveningReadsState) -> list[Send] | str:
    """Fan out to parallel content fetching for each deep-dive.

    Each Send() creates a parallel execution with the assignment details.
    """
    if state.get("status") == "failed":
        return END

    assignments = state.get("deep_dive_assignments", [])
    citation_mappings = state.get("citation_mappings", {})

    if not assignments:
        return END

    sends = []
    for assignment in assignments:
        sends.append(
            Send(
                "fetch_content",
                {
                    "deep_dive_id": assignment["id"],
                    "anchor_keys": assignment["anchor_keys"],
                    "citation_mappings": citation_mappings,
                },
            )
        )

    return sends


def route_to_write(state: EveningReadsState) -> list[Send] | str:
    """Fan out to parallel deep-dive writing.

    This is called from the sync_before_write node, which ensures all
    fetches have completed before we fan out to writes.

    Each writer gets:
    - Its assignment details
    - Its fetched content
    - A list of themes to avoid (from other deep-dives)
    """
    assignments = state.get("deep_dive_assignments", [])
    enriched_content = state.get("enriched_content", [])
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")

    if not assignments:
        return END

    # Build must_avoid lists for distinctiveness
    # Each deep-dive should avoid the themes of the other two
    themes_by_id = {a["id"]: a["theme"] for a in assignments}

    sends = []
    for assignment in assignments:
        # Themes to avoid = all themes except this one
        must_avoid = [
            f"{other_id}: {theme}"
            for other_id, theme in themes_by_id.items()
            if other_id != assignment["id"]
        ]

        # Filter enriched content for this deep-dive
        dd_content = [
            ec for ec in enriched_content if ec["deep_dive_id"] == assignment["id"]
        ]

        sends.append(
            Send(
                "write_deep_dive",
                {
                    "deep_dive_id": assignment["id"],
                    "title": assignment["title"],
                    "theme": assignment["theme"],
                    "structural_approach": assignment["structural_approach"],
                    "anchor_keys": assignment["anchor_keys"],
                    "relevant_sections": assignment["relevant_sections"],
                    "must_avoid": must_avoid,
                    "enriched_content": dd_content,
                    "literature_review": lit_review,
                    "editorial_stance": editorial_stance,
                },
            )
        )

    return sends


async def sync_before_write_node(state: EveningReadsState) -> dict[str, Any]:
    """Synchronization node that waits for all fetches to complete.

    This is a pass-through node that acts as a barrier between the parallel
    fetch operations and the parallel write operations. It ensures all
    enriched_content is collected before fanning out to writers.
    """
    # Just pass through - the state already has merged enriched_content
    enriched = state.get("enriched_content", [])
    import logging

    logging.getLogger(__name__).info(
        f"All fetches complete. Enriched content: {len(enriched)} items"
    )
    return {}


def create_evening_reads_graph() -> StateGraph:
    """Create the workflow graph.

    Flow:
        START -> validate_input
              -> plan_content
              -> [3 parallel fetch_content via Send()]
              -> sync_before_write (barrier)
              -> [3 parallel write_deep_dive via Send()]
              -> write_overview
              -> generate_images (parallel image generation for all 4 articles)
              -> format_references
              -> END

    The fetch and write nodes run in parallel using Send(),
    then their outputs are aggregated via the add reducer on
    enriched_content and deep_dive_drafts in EveningReadsState.

    The sync_before_write node acts as a barrier between the two
    parallel phases, ensuring all fetches complete before writes begin.
    """
    builder = StateGraph(EveningReadsState)

    # Add nodes
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("plan_content", plan_content_node)
    builder.add_node("fetch_content", fetch_content_node)
    builder.add_node("sync_before_write", sync_before_write_node)
    builder.add_node("write_deep_dive", write_deep_dive_node)
    builder.add_node("write_overview", write_overview_node)
    builder.add_node("generate_images", generate_images_node)
    builder.add_node("format_references", format_references_node)

    # Entry point
    builder.add_edge(START, "validate_input")

    # Conditional routing after validation
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        ["plan_content", END],
    )

    # Conditional fan-out to fetch after planning
    builder.add_conditional_edges(
        "plan_content",
        route_to_fetch,
        ["fetch_content", END],
    )

    # All fetch nodes converge to sync node
    builder.add_edge("fetch_content", "sync_before_write")

    # Sync node fans out to writes
    builder.add_conditional_edges(
        "sync_before_write",
        route_to_write,
        ["write_deep_dive", END],
    )

    # All write_deep_dive nodes converge to write_overview
    builder.add_edge("write_deep_dive", "write_overview")

    # Linear flow after overview: overview -> images -> references -> END
    builder.add_edge("write_overview", "generate_images")
    builder.add_edge("generate_images", "format_references")
    builder.add_edge("format_references", END)

    return builder.compile()


# Export compiled graph
evening_reads_graph = create_evening_reads_graph()
