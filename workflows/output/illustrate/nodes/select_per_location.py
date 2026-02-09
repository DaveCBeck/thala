"""Vision pair comparison to select the best candidate at each location."""

import logging

from workflows.shared.vision_comparison import vision_pair_select

from ..state import LocationSelection

logger = logging.getLogger(__name__)


async def select_per_location_node(state: dict) -> dict:
    """Select the best image candidate at a location using vision pair comparison.

    Handles three cases:
    - 0 successful candidates → quality_tier="failed"
    - 1 successful candidate → auto-select, quality_tier="acceptable"
    - 2 successful candidates → vision pair comparison, quality_tier="excellent"

    Args:
        state: Contains location_id, candidates (successful ImageGenResults),
               and selection_criteria for vision comparison

    Returns:
        State update with selection_results
    """
    location_id: str = state["location_id"]
    candidates: list[dict] = state["candidates"]  # successful ImageGenResults
    selection_criteria: str = state.get("selection_criteria", "")

    if not candidates:
        logger.info(f"No successful candidates for {location_id}")
        return {
            "selection_results": [
                LocationSelection(
                    location_id=location_id,
                    selected_brief_id=None,
                    quality_tier="failed",
                    reasoning="Both candidates failed generation",
                )
            ]
        }

    if len(candidates) == 1:
        logger.info(f"Auto-selecting single candidate for {location_id}")
        return {
            "selection_results": [
                LocationSelection(
                    location_id=location_id,
                    selected_brief_id=candidates[0]["brief_id"],
                    quality_tier="acceptable",
                    reasoning="Auto-selected: only one candidate succeeded",
                )
            ]
        }

    # Vision pair comparison
    try:
        png_list = [c["image_bytes"] for c in candidates]
        best_idx = await vision_pair_select(
            png_list,
            selection_criteria=selection_criteria,
        )

        selected = candidates[best_idx]
        logger.info(
            f"Vision pair comparison for {location_id}: "
            f"selected {selected['brief_id']} (candidate {best_idx + 1} of {len(candidates)})"
        )

        return {
            "selection_results": [
                LocationSelection(
                    location_id=location_id,
                    selected_brief_id=selected["brief_id"],
                    quality_tier="excellent",
                    reasoning="Selected via vision pair comparison",
                )
            ]
        }

    except Exception as e:
        # Fallback to first candidate on comparison failure
        logger.warning(f"Vision pair comparison failed for {location_id}: {e}, using first candidate")
        return {
            "selection_results": [
                LocationSelection(
                    location_id=location_id,
                    selected_brief_id=candidates[0]["brief_id"],
                    quality_tier="acceptable",
                    reasoning=f"Vision comparison failed ({e}), auto-selected first candidate",
                )
            ]
        }
