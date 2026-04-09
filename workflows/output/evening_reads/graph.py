"""LangGraph construction for evening_reads workflow.

Transforms academic literature reviews into a 4-part series:
1 overview + 3 deep-dives through:
1. Input validation and citation mapping
2. Content planning with structured output
3. Parallel content fetching (3x) + right-now hook search (3x, recency-high only)
4. Sync barrier + optional replan if hooks missing
5. Parallel deep-dive writing (3x)
6. Overview writing
7. Reference formatting

Illustration is handled separately by the illustrate_and_export task queue workflow.
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .state import EveningReadsState
from .nodes import (
    validate_input_node,
    plan_content_node,
    fetch_content_node,
    find_right_now_node,
    replan_content_node,
    write_deep_dive_node,
    write_overview_node,
    format_references_node,
)

logger = logging.getLogger(__name__)


def route_after_validation(state: EveningReadsState) -> str:
    """Route to planning or end on validation failure."""
    if not state.get("is_valid"):
        return END
    return "plan_content"


def route_to_fetch(state: EveningReadsState) -> list[Send] | str:
    """Fan out to parallel content fetching (and right-now search for recency-high pubs).

    Emits 3 fetch_content Send() calls always.
    For recency-high publications, also emits 3 find_right_now Send() calls
    that run in parallel with the fetches.
    """
    if state.get("status") == "failed":
        return END

    assignments = state.get("deep_dive_assignments", [])
    citation_mappings = state.get("citation_mappings", {})

    if not assignments:
        return END

    editorial_emphasis = state["input"].get("editorial_emphasis", {})
    wants_recency = editorial_emphasis.get("recency") == "high"

    sends = []
    for assignment in assignments:
        # Always fetch content
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

        # For recency-high publications, also search for right-now hooks
        if wants_recency:
            sends.append(
                Send(
                    "find_right_now",
                    {
                        "deep_dive_id": assignment["id"],
                        "title": assignment["title"],
                        "theme": assignment["theme"],
                    },
                )
            )

    return sends


def route_after_sync(state: EveningReadsState) -> list[Send] | str:
    """Route after sync: check if replanning is needed due to missing hooks.

    For recency-high publications, if any deep-dive has zero hooks AND
    we haven't replanned yet, route to replan. Otherwise fan out to writes.
    """
    editorial_emphasis = state["input"].get("editorial_emphasis", {})
    wants_recency = editorial_emphasis.get("recency") == "high"
    replan_attempts = state.get("replan_attempts", 0)

    should_replan = False
    if wants_recency and replan_attempts < 1:
        assignments = state.get("deep_dive_assignments", [])
        right_now_hooks = state.get("right_now_hooks", [])

        hooks_by_id: dict[str, int] = {a["id"]: 0 for a in assignments}
        for hook in right_now_hooks:
            dd_id = hook.get("deep_dive_id")
            if dd_id in hooks_by_id:
                hooks_by_id[dd_id] += 1

        hookless = [dd_id for dd_id, count in hooks_by_id.items() if count == 0]
        if hookless:
            logger.info(f"Deep-dives with zero hooks: {hookless}. Routing to replan.")
            should_replan = True

    if should_replan:
        return "replan_content"

    # No replan needed — fan out to writes
    return route_to_write(state)


def route_to_write(state: EveningReadsState) -> list[Send] | str:
    """Fan out to parallel deep-dive writing.

    Each writer gets its assignment details, fetched content, hooks,
    and a list of themes to avoid (from other deep-dives).
    """
    assignments = state.get("deep_dive_assignments", [])
    enriched_content = state.get("enriched_content", [])
    right_now_hooks = state.get("right_now_hooks", [])
    citation_mappings = state.get("citation_mappings", {})
    lit_review = state["input"]["literature_review"]
    editorial_stance = state["input"].get("editorial_stance", "")
    editorial_emphasis = state["input"].get("editorial_emphasis", {})

    if not assignments:
        return END

    # Build must_avoid lists for distinctiveness
    themes_by_id = {a["id"]: a["theme"] for a in assignments}

    sends = []
    for assignment in assignments:
        must_avoid = [
            f"{other_id}: {theme}" for other_id, theme in themes_by_id.items() if other_id != assignment["id"]
        ]

        # Filter enriched content and hooks for this deep-dive
        dd_content = [ec for ec in enriched_content if ec["deep_dive_id"] == assignment["id"]]
        dd_hooks = [h for h in right_now_hooks if h["deep_dive_id"] == assignment["id"]]

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
                    "right_now_hooks": dd_hooks,
                    "literature_review": lit_review,
                    "editorial_stance": editorial_stance,
                    "editorial_emphasis": editorial_emphasis,
                    "citation_mappings": citation_mappings,
                },
            )
        )

    return sends


async def sync_before_write_node(state: EveningReadsState) -> dict[str, Any]:
    """Synchronization barrier between fetch/find and write phases."""
    enriched = state.get("enriched_content", [])
    hooks = state.get("right_now_hooks", [])
    logger.info(
        f"All fetches complete. Enriched content: {len(enriched)} items, "
        f"right-now hooks: {len(hooks)} items"
    )
    return {}


def create_evening_reads_graph() -> StateGraph:
    """Create the workflow graph.

    Flow:
        START -> validate_input
              -> plan_content
              -> [3x fetch_content + 3x find_right_now via Send()]
              -> sync_before_write (barrier)
              -> route_after_sync:
                  - all have hooks OR already replanned → route_to_write
                  - some missing hooks → replan_content → route_to_fetch (loop)
              -> [3x write_deep_dive via Send()]
              -> write_overview
              -> format_references
              -> END

    The replan loop runs at most once. If the planner can't find a topic
    with recent hooks, it proceeds with the original topic.
    """
    builder = StateGraph(EveningReadsState)

    # Add nodes
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("plan_content", plan_content_node)
    builder.add_node("fetch_content", fetch_content_node)
    builder.add_node("find_right_now", find_right_now_node)
    builder.add_node("sync_before_write", sync_before_write_node)
    builder.add_node("replan_content", replan_content_node)
    builder.add_node("write_deep_dive", write_deep_dive_node)
    builder.add_node("write_overview", write_overview_node)
    builder.add_node("format_references", format_references_node)

    # Entry point
    builder.add_edge(START, "validate_input")

    # Conditional routing after validation
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        ["plan_content", END],
    )

    # Fan-out to fetch (and find_right_now for recency-high) after planning
    builder.add_conditional_edges(
        "plan_content",
        route_to_fetch,
        ["fetch_content", "find_right_now", END],
    )

    # All fetch and find_right_now nodes converge to sync node
    builder.add_edge("fetch_content", "sync_before_write")
    builder.add_edge("find_right_now", "sync_before_write")

    # After sync: check hooks, maybe replan or fan out to writes
    builder.add_conditional_edges(
        "sync_before_write",
        route_after_sync,
        ["write_deep_dive", "replan_content"],
    )

    # Replan loops back to fetch + find_right_now
    builder.add_conditional_edges(
        "replan_content",
        route_to_fetch,
        ["fetch_content", "find_right_now", END],
    )

    # All write_deep_dive nodes converge to write_overview
    builder.add_edge("write_deep_dive", "write_overview")

    # Linear flow after overview
    builder.add_edge("write_overview", "format_references")
    builder.add_edge("format_references", END)

    return builder.compile()


# Export compiled graph
evening_reads_graph = create_evening_reads_graph()
