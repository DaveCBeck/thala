"""Assemble full document with images for editorial review.

Pure Python node (no LLM call) that inserts all winning images into the
markdown as base64 data URIs so the editorial review vision model can see
the complete illustrated document.
"""

import base64
import logging
import re

from ..schemas import ImageLocationPlan
from ..state import AssembledImage, IllustrateState
from .finalize import _select_winning_results

logger = logging.getLogger(__name__)


def _build_assembled_markdown(
    document: str,
    winners_with_meta: list[tuple[dict, ImageLocationPlan]],
) -> str:
    """Insert base64 data URI images into markdown below specified headers.

    Similar to finalize._insert_images_into_markdown but uses inline
    base64 data URIs instead of file paths.
    """
    if not winners_with_meta:
        return document

    lines = document.split("\n")
    insertions: list[tuple[int, str]] = []

    for gen_result, plan in winners_with_meta:
        image_bytes = gen_result.get("image_bytes")
        if not image_bytes:
            continue

        header_text = plan.insertion_after_header
        header_line_idx = None
        for i, line in enumerate(lines):
            if re.match(r"^#+\s*" + re.escape(header_text) + r"\s*$", line.strip()):
                header_line_idx = i
                break
            if header_text in line and line.strip().startswith("#"):
                header_line_idx = i
                break

        if header_line_idx is None:
            logger.warning(f"Could not find header '{header_text}' for {plan.location_id}")
            continue

        # Detect media type from magic bytes
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            media_type = "image/png"
        else:
            media_type = "image/jpeg"

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        alt_text = gen_result.get("alt_text") or f"Image: {header_text}"
        image_md = f"\n![{alt_text}](data:{media_type};base64,{b64})\n"

        insertions.append((header_line_idx + 1, image_md))

    insertions.sort(key=lambda x: x[0], reverse=True)
    for line_idx, md in insertions:
        lines.insert(line_idx, md)

    return "\n".join(lines)


async def assemble_document_node(state: IllustrateState) -> dict:
    """Insert all selected images into markdown for editorial review.

    This is a temporary assembly — the editorial review may remove some.
    Images are base64-encoded inline so the vision model can see the full
    illustrated document.

    Returns:
        State update with assembled_document and assembled_images.
    """
    generation_results = state.get("generation_results", [])
    selection_results = state.get("selection_results", [])
    image_plan = state.get("image_plan", [])
    image_opportunities = state.get("image_opportunities", [])
    document = state["input"]["markdown_document"]

    # Pick winning results (same logic as finalize)
    winners = _select_winning_results(generation_results, selection_results)

    # Build plan lookup
    plans_by_id = {p.location_id: p for p in image_plan}

    # Build opportunity lookup for purpose
    purpose_by_id = {opp.location_id: opp.purpose for opp in image_opportunities}

    # Pair winners with their plans and build metadata
    winners_with_meta: list[tuple[dict, ImageLocationPlan]] = []
    assembled_images: list[AssembledImage] = []

    for gen_result in winners:
        location_id = gen_result["location_id"]
        plan = plans_by_id.get(location_id)
        if not plan or not gen_result.get("image_bytes"):
            continue

        winners_with_meta.append((gen_result, plan))
        assembled_images.append(
            AssembledImage(
                location_id=location_id,
                image_type=gen_result["image_type"],
                purpose=purpose_by_id.get(location_id, plan.purpose),
                image_bytes=gen_result["image_bytes"],
            )
        )

    # Build markdown with inline base64 images
    assembled_document = _build_assembled_markdown(document, winners_with_meta)

    logger.info(f"Assembled document with {len(assembled_images)} images")

    return {
        "assembled_document": assembled_document,
        "assembled_images": assembled_images,
    }
