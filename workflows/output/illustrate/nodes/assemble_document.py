"""Assemble image metadata for editorial review.

Pure Python node (no LLM call) that collects all winning images into
AssembledImage records so the editorial review vision model can evaluate
the full illustrated set.
"""

import logging

from ..state import AssembledImage, IllustrateState
from ..utils import select_winning_results

logger = logging.getLogger(__name__)


async def assemble_document_node(state: IllustrateState) -> dict:
    """Collect all selected images for editorial review.

    This is a temporary assembly -- the editorial review may remove some.

    Returns:
        State update with assembled_images.
    """
    generation_results = state.get("generation_results", [])
    selection_results = state.get("selection_results", [])
    image_plan = state.get("image_plan", [])
    image_opportunities = state.get("image_opportunities", [])

    # Pick winning results (same logic as finalize)
    winners = select_winning_results(generation_results, selection_results)

    # Build plan lookup
    plans_by_id = {p.location_id: p for p in image_plan}

    # Build opportunity lookup for purpose
    purpose_by_id = {opp.location_id: opp.purpose for opp in image_opportunities}

    # Pair winners with their plans and build metadata
    assembled_images: list[AssembledImage] = []

    for gen_result in winners:
        location_id = gen_result["location_id"]
        plan = plans_by_id.get(location_id)
        if not plan or not gen_result.get("image_bytes"):
            continue

        assembled_images.append(
            AssembledImage(
                location_id=location_id,
                image_type=gen_result["image_type"],
                purpose=purpose_by_id.get(location_id, plan.purpose),
                image_bytes=gen_result["image_bytes"],
            )
        )

    logger.info(f"Assembled {len(assembled_images)} images for editorial review")

    return {
        "assembled_images": assembled_images,
    }
