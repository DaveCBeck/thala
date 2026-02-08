"""Finalize workflow: save images and insert into document."""

import base64
import logging
import os
import re
import tempfile
from collections import defaultdict
from typing import Literal

from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from ..config import IllustrateConfig
from ..prompts import VISION_COMPARE_SYSTEM, VISION_COMPARE_USER
from ..schemas import ImageCompareResult, ImageLocationPlan
from ..state import (
    FinalImage,
    ImageGenResult,
    ImageReviewResult,
    IllustrateState,
    WorkflowError,
)

logger = logging.getLogger(__name__)


async def _compare_candidates(
    candidates: list[ImageGenResult],
    document_context: str,
    purpose: str,
    brief: str,
) -> ImageGenResult:
    """Compare multiple image candidates and select the best one.

    Uses vision to evaluate all candidates side-by-side and pick the best.
    Falls back to the last candidate if comparison fails.
    """
    if len(candidates) == 1:
        return candidates[0]

    try:
        # Build vision content with all candidate images
        content = [
            {
                "type": "text",
                "text": VISION_COMPARE_USER.format(
                    num_candidates=len(candidates),
                    context=document_context[:3000],
                    purpose=purpose,
                    brief=brief[:1500],
                ),
            }
        ]

        # Add each candidate image
        for i, candidate in enumerate(candidates):
            if not candidate["image_bytes"]:
                continue

            image_bytes = candidate["image_bytes"]
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Determine media type
            media_type = "image/png"
            if image_bytes[:2] == b"\xff\xd8":
                media_type = "image/jpeg"

            content.append(
                {
                    "type": "text",
                    "text": f"\n**Candidate {i + 1}:**",
                }
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_image,
                    },
                }
            )

        result = await invoke(
            tier=ModelTier.SONNET,
            system=VISION_COMPARE_SYSTEM,
            user=content,
            schema=ImageCompareResult,
            config=InvokeConfig(max_tokens=500),
        )

        # Get the selected candidate (1-indexed)
        selected_idx = result.selected_candidate - 1
        if 0 <= selected_idx < len(candidates):
            logger.info(f"Comparison selected candidate {result.selected_candidate}: {result.reasoning[:100]}")
            if result.issues_with_selected:
                logger.warning(f"Selected image has issues: {result.issues_with_selected}")
            return candidates[selected_idx]
        else:
            logger.warning(f"Invalid candidate index {result.selected_candidate}, using last")
            return candidates[-1]

    except Exception as e:
        logger.warning(f"Comparison failed, using last candidate: {e}")
        return candidates[-1]


def _group_candidates_by_location(
    generation_results: list[ImageGenResult],
    review_results: list[ImageReviewResult],
) -> tuple[dict[str, list[ImageGenResult]], set[str]]:
    """Group successful generation results by location_id.

    Returns:
        - Dict mapping location_id to list of ALL successful candidates
        - Set of location_ids that have at least one passed review
    """
    # Build set of passed location_ids
    passed_ids = {r["location_id"] for r in review_results if r.get("passed", False)}

    # Group ALL successful generations by location_id
    candidates_by_id: dict[str, list[ImageGenResult]] = defaultdict(list)
    for gen in generation_results:
        if not gen["success"] or not gen.get("image_bytes"):
            continue
        candidates_by_id[gen["location_id"]].append(gen)

    return dict(candidates_by_id), passed_ids


def _insert_images_into_markdown(
    document: str,
    final_images: list[FinalImage],
    image_plan: list[ImageLocationPlan],
) -> str:
    """Insert image references into markdown below specified headers.

    Images are inserted as: ![alt](path)

    Attributions are added as small captions below images when required.
    """
    if not final_images:
        return document

    # Build lookup of images by location_id
    images_by_id = {img["location_id"]: img for img in final_images}

    # Process each plan in reverse order (so line numbers stay valid)
    lines = document.split("\n")
    insertions = []  # List of (line_index, markdown_to_insert)

    for plan in image_plan:
        if plan.location_id not in images_by_id:
            continue

        img = images_by_id[plan.location_id]
        header_text = plan.insertion_after_header

        # Find the header line
        header_line_idx = None
        for i, line in enumerate(lines):
            # Match header with varying levels (# ## ### etc.)
            if re.match(r"^#+\s*" + re.escape(header_text) + r"\s*$", line.strip()):
                header_line_idx = i
                break
            # Also try partial match for long headers
            if header_text in line and line.strip().startswith("#"):
                header_line_idx = i
                break

        if header_line_idx is None:
            logger.warning(f"Could not find header '{header_text}' for image {plan.location_id}")
            continue

        # Build image markdown
        alt_text = img["alt_text"] or f"Image: {header_text}"
        file_path = img["file_path"]
        image_md = f"\n![{alt_text}]({file_path})\n"

        # Add source attribution based on image type
        image_type = img.get("image_type", "generated")
        if image_type == "public_domain" and img.get("attribution"):
            attr = img["attribution"]
            photographer = attr.get("photographer", "Unknown")
            source = attr.get("source", "")
            source_url = attr.get("source_url", "")
            if source_url:
                image_md += f"\n*Photo by {photographer} via [{source}]({source_url})*\n"
            else:
                image_md += f"\n*Photo by {photographer} via {source}*\n"
        elif image_type == "diagram":
            image_md += "\n*Diagram generated by Claude*\n"
        elif image_type == "generated":
            image_md += "\n*Image generated by Imagen*\n"

        # Insert after the header line
        insertions.append((header_line_idx + 1, image_md))

    # Sort by line index descending and insert
    insertions.sort(key=lambda x: x[0], reverse=True)
    for line_idx, md in insertions:
        lines.insert(line_idx, md)

    return "\n".join(lines)


def _determine_status(
    image_plan: list[ImageLocationPlan],
    final_images: list[FinalImage],
    errors: list[WorkflowError],
) -> Literal["success", "partial", "failed"]:
    """Determine workflow status based on results."""
    if not image_plan:
        return "failed"

    if not final_images:
        return "failed"

    if len(final_images) == len(image_plan):
        return "success"

    # Some images succeeded
    return "partial"


async def finalize_node(state: IllustrateState) -> dict:
    """Finalize workflow: save images to files and insert into markdown.

    This node:
    1. Filters to approved images (passed review)
    2. Saves each image to the output directory
    3. Inserts image references into the markdown
    4. Returns the final illustrated document

    Returns:
        State update with final_images, illustrated_document, status
    """
    config = state.get("config") or IllustrateConfig()
    document = state["input"]["markdown_document"]
    image_plan = state.get("image_plan", [])
    generation_results = state.get("generation_results", [])
    review_results = state.get("review_results", [])
    existing_errors = state.get("errors", [])

    # Get output directory
    output_dir = config.output_dir or state["input"].get("output_dir")
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="illustrate_")
        logger.info(f"Using temp directory for images: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Build lookup for plans (needed for comparison and saving)
    plans_by_id = {p.location_id: p for p in image_plan}

    # Group ALL candidates by location
    candidates_by_id, passed_ids = _group_candidates_by_location(generation_results, review_results)

    # For each location, either use the passed candidate or compare all candidates
    approved: list[ImageGenResult] = []
    for loc_id, candidates in candidates_by_id.items():
        if loc_id in passed_ids:
            # At least one passed - use the last one that passed
            # (most recent successful attempt)
            approved.append(candidates[-1])
        elif len(candidates) == 1:
            # Only one candidate, nothing passed - use it anyway with warning
            logger.warning(f"Using only candidate for {loc_id} despite review issues")
            approved.append(candidates[0])
        else:
            # Multiple candidates, none passed - compare and select best
            plan = plans_by_id.get(loc_id)
            if plan:
                logger.warning(
                    f"No candidate passed review for {loc_id}, comparing {len(candidates)} candidates to select best"
                )
                best = await _compare_candidates(
                    candidates=candidates,
                    document_context=document[:3000],
                    purpose=plan.purpose,
                    brief=plan.brief,
                )
                approved.append(best)
            else:
                # No plan, just use last candidate
                logger.warning(f"No plan for {loc_id}, using last candidate")
                approved.append(candidates[-1])

    logger.info(f"Finalizing {len(approved)} approved images")

    # Save images and build final_images list
    final_images: list[FinalImage] = []
    errors: list[WorkflowError] = []

    for gen_result in approved:
        location_id = gen_result["location_id"]
        plan = plans_by_id.get(location_id)

        if not plan:
            logger.warning(f"No plan found for {location_id}")
            continue

        if not gen_result["image_bytes"]:
            logger.warning(f"No image bytes for {location_id}")
            continue

        # Determine file extension
        image_type = gen_result["image_type"]
        ext = "png"  # Default for diagrams and Imagen
        if image_type == "public_domain":
            # Check if JPEG
            if gen_result["image_bytes"][:2] == b"\xff\xd8":
                ext = "jpg"

        # Save file
        filename = f"{location_id}.{ext}"
        file_path = os.path.join(output_dir, filename)

        try:
            with open(file_path, "wb") as f:
                f.write(gen_result["image_bytes"])
            logger.info(f"Saved image: {file_path}")

            final_images.append(
                FinalImage(
                    location_id=location_id,
                    insertion_after_header=plan.insertion_after_header,
                    file_path=file_path,
                    alt_text=gen_result["alt_text"] or f"Image: {plan.insertion_after_header}",
                    image_type=image_type,
                    attribution=gen_result["attribution"],
                )
            )

        except Exception as e:
            logger.error(f"Failed to save image {location_id}: {e}")
            errors.append(
                WorkflowError(
                    location_id=location_id,
                    severity="error",
                    message=f"Failed to save image: {e}",
                    stage="finalize",
                )
            )

    # Insert images into document
    illustrated_document = _insert_images_into_markdown(
        document=document,
        final_images=final_images,
        image_plan=image_plan,
    )

    # Determine status
    all_errors = existing_errors + errors
    status = _determine_status(image_plan, final_images, all_errors)

    # Log which images were dropped (failed review after retries)
    planned_ids = {p.location_id for p in image_plan}
    approved_ids = {img["location_id"] for img in final_images}
    dropped_ids = planned_ids - approved_ids
    for loc_id in dropped_ids:
        logger.warning(f"Image dropped (failed review after retries): {loc_id}")

    logger.info(f"Finalize complete: {len(final_images)}/{len(image_plan)} images, status={status}")

    return {
        "final_images": final_images,
        "illustrated_document": illustrated_document,
        "errors": errors,
        "status": status,
    }
