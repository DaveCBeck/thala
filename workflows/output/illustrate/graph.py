"""LangGraph construction for illustrate workflow.

Graph shape:
    START
      ↓
    creative_direction          (Pass 1: visual identity + opportunity map)
      ↓
    plan_briefs                 (Pass 2: candidate briefs per location)
      ↓
    [fan-out] generate_candidate  (one per brief, ~12 parallel tasks)
      ↓
    sync_after_generation       (barrier — counts results)
      ↓
    [fan-out] select_per_location (one per location, ~6 parallel tasks)
      ↓
    sync_after_selection        (barrier)
      ↓
    [conditional] retry_failed  (only locations where both failed)
      ↓
    finalize                    (save files, insert into markdown)
      ↓
    END
"""

import logging
from collections import defaultdict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import IllustrateConfig
from .nodes import (
    creative_direction_node,
    finalize_node,
    generate_candidate_node,
    plan_briefs_node,
    select_per_location_node,
)
from .schemas import CandidateBrief, ImageLocationPlan
from .state import IllustrateState

logger = logging.getLogger(__name__)

# Cross-strategy fallback: when both candidates fail, switch image source type
_FALLBACK_IMAGE_TYPE = {
    "public_domain": "generated",
    "generated": "public_domain",
    "diagram": "generated",
}


def _find_plan_by_id(
    image_plan: list[ImageLocationPlan],
    location_id: str,
) -> ImageLocationPlan | None:
    """Find plan by location_id."""
    for plan in image_plan:
        if plan.location_id == location_id:
            return plan
    return None


def route_after_analysis(state: IllustrateState) -> list[Send] | str:
    """Fan out one Send per CandidateBrief for parallel generation.

    If analysis failed or no briefs, go to finalize.
    """
    if state.get("status") == "failed":
        logger.info("Analysis failed, going to finalize")
        return "finalize"

    candidate_briefs = state.get("candidate_briefs", [])
    image_plan = state.get("image_plan", [])

    if not candidate_briefs or not image_plan:
        logger.info("No briefs or plans, going to finalize")
        return "finalize"

    config = state.get("config") or IllustrateConfig()
    document = state["input"]["markdown_document"]
    visual_identity = state.get("visual_identity")

    sends = []
    for brief in candidate_briefs:
        plan = _find_plan_by_id(image_plan, brief.location_id)
        if not plan:
            continue

        brief_id = f"{brief.location_id}_{brief.candidate_index}"
        send_data = {
            "location": plan,
            "brief": brief,
            "brief_id": brief_id,
            "document_context": document,
            "config": config,
            "visual_identity": visual_identity,
        }
        sends.append(Send("generate_candidate", send_data))

    logger.info(f"Routing to {len(sends)} candidate generation nodes")
    return sends or "finalize"


def sync_after_generation(state: IllustrateState) -> dict:
    """Synchronization barrier after all generations complete."""
    generation_results = state.get("generation_results", [])
    successful = sum(1 for r in generation_results if r["success"])
    logger.info(f"Generation sync: {successful}/{len(generation_results)} successful")
    return {}


def route_to_selection(state: IllustrateState) -> list[Send] | str:
    """Group generation results by location_id, fan out pair comparison."""
    generation_results = state.get("generation_results", [])
    candidate_briefs = state.get("candidate_briefs", [])

    if not generation_results:
        logger.info("No generation results, going to finalize")
        return "finalize"

    # Group by location_id
    by_location: dict[str, list[dict]] = defaultdict(list)
    for r in generation_results:
        by_location[r["location_id"]].append(r)

    # Build brief text lookup for selection criteria
    brief_text_by_location: dict[str, str] = {}
    for brief in candidate_briefs:
        if brief.location_id not in brief_text_by_location:
            brief_text_by_location[brief.location_id] = brief.brief

    sends = []
    for location_id, candidates in by_location.items():
        successful = [c for c in candidates if c["success"] and c.get("image_bytes")]
        sends.append(
            Send(
                "select_per_location",
                {
                    "location_id": location_id,
                    "candidates": successful,
                    "selection_criteria": brief_text_by_location.get(location_id, ""),
                },
            )
        )

    logger.info(f"Routing to {len(sends)} selection nodes")
    return sends or "finalize"


def sync_after_selection(state: IllustrateState) -> dict:
    """Synchronization barrier after all selections complete.

    Updates retry_count for failed locations.
    """
    selection_results = state.get("selection_results", [])
    passed = sum(1 for s in selection_results if s["quality_tier"] != "failed")
    failed = sum(1 for s in selection_results if s["quality_tier"] == "failed")

    # Update retry counts for failed locations
    retry_count = dict(state.get("retry_count", {}))
    for s in selection_results:
        if s["quality_tier"] == "failed":
            retry_count[s["location_id"]] = retry_count.get(s["location_id"], 0) + 1

    logger.info(f"Selection sync: {passed} selected, {failed} failed")
    return {"retry_count": retry_count}


def route_after_selection(state: IllustrateState) -> list[Send] | str:
    """Route failed locations to retry with cross-strategy fallback.

    Generates two retry candidates per failed location, switching image source type.
    """
    config = state.get("config") or IllustrateConfig()
    selection_results = state.get("selection_results", [])
    retry_count = state.get("retry_count", {})
    image_plan = state.get("image_plan", [])
    candidate_briefs = state.get("candidate_briefs", [])
    document = state["input"]["markdown_document"]
    visual_identity = state.get("visual_identity")

    # Find failed locations eligible for retry
    failed_locations = [
        s["location_id"]
        for s in selection_results
        if s["quality_tier"] == "failed" and retry_count.get(s["location_id"], 0) <= config.max_retries
    ]

    if not failed_locations:
        logger.info("No failed locations to retry, going to finalize")
        return "finalize"

    # Build brief lookup
    briefs_by_location: dict[str, list[CandidateBrief]] = defaultdict(list)
    for brief in candidate_briefs:
        briefs_by_location[brief.location_id].append(brief)

    sends = []
    for loc_id in failed_locations:
        plan = _find_plan_by_id(image_plan, loc_id)
        if not plan:
            continue

        location_briefs = briefs_by_location.get(loc_id, [])
        if not location_briefs:
            continue

        # Generate two retry candidates with fallback image types
        for i, orig_brief in enumerate(location_briefs[:2]):
            fallback_type = _FALLBACK_IMAGE_TYPE.get(orig_brief.image_type, "generated")
            retry_brief = CandidateBrief(
                location_id=orig_brief.location_id,
                candidate_index=orig_brief.candidate_index,
                image_type=fallback_type,
                brief=orig_brief.brief,
                relationship_to_text=orig_brief.relationship_to_text,
                visual_identity_references=orig_brief.visual_identity_references,
                literal_queries=orig_brief.literal_queries,
                conceptual_queries=orig_brief.conceptual_queries,
                query_strategy=orig_brief.query_strategy,
                diagram_subtype=orig_brief.diagram_subtype,
            )

            brief_id = f"{loc_id}_{orig_brief.candidate_index}_retry"
            send_data = {
                "location": plan,
                "brief": retry_brief,
                "brief_id": brief_id,
                "document_context": document,
                "config": config,
                "visual_identity": visual_identity,
            }
            sends.append(Send("generate_candidate", send_data))

    if sends:
        logger.info(f"Routing to {len(sends)} retry generation nodes")
        return sends

    return "finalize"


def create_illustrate_graph() -> StateGraph:
    """Create the illustrate workflow graph.

    Flow:
        START → creative_direction → plan_briefs
          → [fan-out] generate_candidate (~12 parallel)
          → sync_after_generation
          → [fan-out] select_per_location (~6 parallel)
          → sync_after_selection
          → [conditional] retry (generate_candidate) or finalize
          → finalize → END
    """
    builder = StateGraph(IllustrateState)

    # Add nodes
    builder.add_node("creative_direction", creative_direction_node)
    builder.add_node("plan_briefs", plan_briefs_node)
    builder.add_node("generate_candidate", generate_candidate_node)
    builder.add_node("sync_after_generation", sync_after_generation)
    builder.add_node("select_per_location", select_per_location_node)
    builder.add_node("sync_after_selection", sync_after_selection)
    builder.add_node("finalize", finalize_node)

    # Two-pass planning
    builder.add_edge(START, "creative_direction")
    builder.add_edge("creative_direction", "plan_briefs")

    # Fan out to generation per brief
    builder.add_conditional_edges(
        "plan_briefs",
        route_after_analysis,
        ["generate_candidate", "finalize"],
    )

    # All generation nodes converge to sync
    builder.add_edge("generate_candidate", "sync_after_generation")

    # After generation sync, fan out to selection per location
    builder.add_conditional_edges(
        "sync_after_generation",
        route_to_selection,
        ["select_per_location", "finalize"],
    )

    # All selection nodes converge to sync
    builder.add_edge("select_per_location", "sync_after_selection")

    # After selection sync, retry failed or finalize
    builder.add_conditional_edges(
        "sync_after_selection",
        route_after_selection,
        ["generate_candidate", "finalize"],
    )

    # Finalize to end
    builder.add_edge("finalize", END)

    return builder.compile()


# Export the compiled graph
illustrate_graph = create_illustrate_graph()
