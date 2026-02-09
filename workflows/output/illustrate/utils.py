"""Shared utilities for the illustrate workflow."""

import logging
from collections import defaultdict

from .state import ImageGenResult, LocationSelection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def detect_media_type(data: bytes) -> str:
    """Detect image media type from magic bytes.

    Returns ``"image/png"`` when the data starts with the PNG signature,
    otherwise falls back to ``"image/jpeg"``.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def select_winning_results(
    generation_results: list[ImageGenResult],
    selection_results: list[LocationSelection],
) -> list[ImageGenResult]:
    """Pick the winning ImageGenResult per location based on selection_results.

    For each location with a selection, find the generation result matching
    the selected_brief_id. Falls back to the last successful result if
    the selected brief_id can't be found.
    """
    # Build lookup: brief_id -> ImageGenResult
    results_by_brief_id: dict[str, ImageGenResult] = {}
    results_by_location: dict[str, list[ImageGenResult]] = defaultdict(list)
    for gen in generation_results:
        if gen["success"] and gen.get("image_bytes"):
            results_by_brief_id[gen["brief_id"]] = gen
            results_by_location[gen["location_id"]].append(gen)

    # Deduplicate selections: keep only the LAST entry per location_id.
    # selection_results uses an `add` reducer, so retry-round entries appear
    # after earlier rounds.  Without dedup the first (often "failed") entry
    # claims the slot and the retry success is silently skipped.
    latest_selection: dict[str, LocationSelection] = {}
    for selection in selection_results:
        latest_selection[selection["location_id"]] = selection

    winners: list[ImageGenResult] = []
    selected_locations: set[str] = set()

    for loc_id, selection in latest_selection.items():
        selected_locations.add(loc_id)

        if selection["quality_tier"] == "failed":
            continue  # Both candidates failed, no image for this location

        brief_id = selection["selected_brief_id"]
        if brief_id and brief_id in results_by_brief_id:
            winners.append(results_by_brief_id[brief_id])
        elif loc_id in results_by_location:
            # Fallback: use the last successful result for this location
            logger.warning(f"Selected brief_id {brief_id} not found for {loc_id}, using fallback")
            winners.append(results_by_location[loc_id][-1])

    # Also include any successful results for locations without selection
    # (e.g., from retry rounds that bypass selection)
    for loc_id, results in results_by_location.items():
        if loc_id not in selected_locations:
            winners.append(results[-1])

    return winners
