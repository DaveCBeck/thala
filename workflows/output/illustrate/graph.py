"""LangGraph construction for illustrate workflow."""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import IllustrateConfig
from .nodes import (
    analyze_document_node,
    finalize_node,
    generate_additional_node,
    generate_header_node,
    review_image_node,
)
from .schemas import ImageLocationPlan
from .state import IllustrateState

logger = logging.getLogger(__name__)


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
    """Route to generation after analysis.

    If analysis failed or no images planned, go to finalize.
    Otherwise, fan out to generate nodes.
    """
    if state.get("status") == "failed":
        logger.info("Analysis failed, going to finalize")
        return "finalize"

    image_plan = state.get("image_plan", [])
    if not image_plan:
        logger.info("No images planned, going to finalize")
        return "finalize"

    config = state.get("config") or IllustrateConfig()
    document = state["input"]["markdown_document"]

    sends = []

    for plan in image_plan:
        # Header uses special node
        if plan.purpose == "header":
            sends.append(
                Send(
                    "generate_header",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                    },
                )
            )
        else:
            sends.append(
                Send(
                    "generate_additional",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                    },
                )
            )

    logger.info(f"Routing to {len(sends)} generation nodes")
    return sends


def sync_after_generation(state: IllustrateState) -> dict:
    """Synchronization barrier after all generations complete."""
    generation_results = state.get("generation_results", [])
    successful = sum(1 for r in generation_results if r["success"])
    logger.info(
        f"Generation sync: {successful}/{len(generation_results)} successful"
    )
    return {}


def route_to_review(state: IllustrateState) -> list[Send] | str:
    """Route to vision review or finalize.

    If vision review is disabled, go directly to finalize.
    Otherwise, fan out review nodes for successful generations.
    """
    config = state.get("config") or IllustrateConfig()

    if not config.enable_vision_review:
        logger.info("Vision review disabled, going to finalize")
        return "finalize"

    generation_results = state.get("generation_results", [])
    successful = [r for r in generation_results if r["success"]]

    if not successful:
        logger.info("No successful generations, going to finalize")
        return "finalize"

    image_plan = state.get("image_plan", [])
    document = state["input"]["markdown_document"]

    sends = []
    for gen_result in successful:
        location_id = gen_result["location_id"]
        plan = _find_plan_by_id(image_plan, location_id)

        if plan:
            sends.append(
                Send(
                    "review_image",
                    {
                        "generation_result": gen_result,
                        "location": plan,
                        "document_context": document,
                    },
                )
            )

    logger.info(f"Routing to {len(sends)} review nodes")
    return sends


def sync_after_review(state: IllustrateState) -> dict:
    """Synchronization barrier and retry preparation after review.

    Updates retry_count for any pending retries.
    """
    review_results = state.get("review_results", [])
    passed = sum(1 for r in review_results if r.get("passed", False))

    # Update retry counts
    pending_retries = state.get("pending_retries", [])
    retry_count = dict(state.get("retry_count", {}))

    for loc_id in pending_retries:
        retry_count[loc_id] = retry_count.get(loc_id, 0) + 1

    logger.info(
        f"Review sync: {passed}/{len(review_results)} passed, "
        f"{len(pending_retries)} pending retries"
    )

    return {"retry_count": retry_count}


def route_after_review(state: IllustrateState) -> list[Send] | str:
    """Route to retry generation or finalize.

    If there are pending retries within the retry limit, fan out.
    Otherwise, go to finalize.
    """
    config = state.get("config") or IllustrateConfig()
    pending_retries = state.get("pending_retries", [])
    retry_count = state.get("retry_count", {})
    retry_briefs = state.get("retry_briefs", {})
    image_plan = state.get("image_plan", [])
    document = state["input"]["markdown_document"]

    # Filter to retries within limit
    eligible_retries = [
        loc_id
        for loc_id in pending_retries
        if retry_count.get(loc_id, 0) <= config.max_retries
    ]

    if not eligible_retries:
        logger.info("No eligible retries, going to finalize")
        return "finalize"

    sends = []
    for loc_id in eligible_retries:
        plan = _find_plan_by_id(image_plan, loc_id)
        if not plan:
            continue

        retry_brief = retry_briefs.get(loc_id)

        # Use appropriate generation node
        if plan.purpose == "header":
            sends.append(
                Send(
                    "generate_header",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                        "is_retry": True,
                        "retry_brief": retry_brief,
                    },
                )
            )
        else:
            sends.append(
                Send(
                    "generate_additional",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                        "is_retry": True,
                        "retry_brief": retry_brief,
                    },
                )
            )

    if sends:
        logger.info(f"Routing to {len(sends)} retry generation nodes")
        return sends

    return "finalize"


def create_illustrate_graph() -> StateGraph:
    """Create the illustrate workflow graph.

    Flow:
        START
          ↓
        analyze_document (plan all image locations)
          ↓
        [conditional] → generate_header / generate_additional (parallel via Send)
          ↓
        sync_after_generation (barrier)
          ↓
        [conditional] → review_image (parallel via Send) or finalize
          ↓
        sync_after_review (barrier)
          ↓
        [conditional] → retry generation or finalize
          ↓
        finalize (save files, insert into markdown)
          ↓
        END
    """
    builder = StateGraph(IllustrateState)

    # Add nodes
    builder.add_node("analyze_document", analyze_document_node)
    builder.add_node("generate_header", generate_header_node)
    builder.add_node("generate_additional", generate_additional_node)
    builder.add_node("sync_after_generation", sync_after_generation)
    builder.add_node("review_image", review_image_node)
    builder.add_node("sync_after_review", sync_after_review)
    builder.add_node("finalize", finalize_node)

    # Entry point
    builder.add_edge(START, "analyze_document")

    # After analysis, fan out to generation
    builder.add_conditional_edges(
        "analyze_document",
        route_after_analysis,
        ["generate_header", "generate_additional", "finalize"],
    )

    # All generation nodes converge to sync
    builder.add_edge("generate_header", "sync_after_generation")
    builder.add_edge("generate_additional", "sync_after_generation")

    # After generation sync, route to review or finalize
    builder.add_conditional_edges(
        "sync_after_generation",
        route_to_review,
        ["review_image", "finalize"],
    )

    # All review nodes converge to sync
    builder.add_edge("review_image", "sync_after_review")

    # After review sync, route to retry or finalize
    builder.add_conditional_edges(
        "sync_after_review",
        route_after_review,
        ["generate_header", "generate_additional", "finalize"],
    )

    # Finalize to end
    builder.add_edge("finalize", END)

    return builder.compile()


# Export the compiled graph
illustrate_graph = create_illustrate_graph()
