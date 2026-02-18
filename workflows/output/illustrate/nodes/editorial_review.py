"""Editorial review: vision-based full-document image curation.

A single SONNET vision call evaluates all non-header images in context and
identifies the weakest for cutting. This is the N-from-N+2 quality gate.
"""

import base64
import logging

from langsmith import traceable

from workflows.shared.llm_utils import ModelTier, invoke, InvokeConfig

from ..schemas import EditorialReviewResult, ImageOpportunity
from ..state import AssembledImage, IllustrateState
from ..utils import detect_media_type
from .generate_additional import MAX_IMAGE_SIZE

logger = logging.getLogger(__name__)

EDITORIAL_SYSTEM = """You are an editorial art director reviewing the full set of illustrations for a long-form article. Your job is to ensure the illustrations work as a cohesive set."""

EDITORIAL_USER = """This article was intentionally illustrated with {cuts_count} more images than needed so you can select the strongest set. You will evaluate {n_images} non-header images and cut the {cuts_count} that contribute least to the overall article.

Evaluate each image on:
1. Visual coherence — Does it match the style/identity of the other images?
2. Pacing contribution — Is it well-placed? Does it avoid clustering?
3. Variety contribution — Is it different from its neighbors? Different type than adjacent?
4. Individual quality — Is it technically good and contextually relevant?

Visual identity for this article:
- Style: {primary_style}
- Palette: {color_palette}
- Mood: {mood}

Rank ALL {n_images} images from strongest to weakest contribution, then mark the bottom {cuts_count} for removal. For each cut, explain why.

The images are shown below with their location IDs."""


def _compute_cuts_count(
    non_header_images: list[AssembledImage],
    image_opportunities: list[ImageOpportunity],
) -> int:
    """Compute how many images to cut.

    Rule: never cut below the target N.
    Target N = total opportunities - 2 (the over-generation surplus).
    Cuts = min(2, non_header_count - target_non_header_count).
    """
    # Count non-header opportunities to derive the target
    non_header_opportunities = [opp for opp in image_opportunities if opp.purpose != "header"]
    # Target non-header count = total non-header opportunities - 2
    target_non_header = max(0, len(non_header_opportunities) - 2)

    surplus = len(non_header_images) - target_non_header
    return max(0, min(2, surplus))


@traceable(run_type="chain", name="Illustrate_EditorialReview")
async def editorial_review_node(state: IllustrateState) -> dict:
    """Vision-based editorial review of the full illustrated document.

    Evaluates all non-header images and identifies the weakest for cutting.
    Uses a single SONNET vision call with all images.

    Returns:
        State update with editorial_review_result.
    """
    assembled_images = state.get("assembled_images", [])
    image_opportunities = state.get("image_opportunities", [])
    visual_identity = state.get("visual_identity")

    # Filter to non-header images only
    non_header_images = [img for img in assembled_images if img["purpose"] != "header"]

    # Compute adaptive cut count
    cuts_count = _compute_cuts_count(non_header_images, image_opportunities)

    if cuts_count == 0 or len(non_header_images) == 0:
        logger.info(f"Editorial review: no cuts needed ({len(non_header_images)} non-header images, {cuts_count} cuts)")
        result = EditorialReviewResult(
            evaluations=[],
            cut_location_ids=[],
            editorial_summary=f"No cuts needed: {len(non_header_images)} non-header images, target met.",
        )
        return {
            "editorial_review_result": result.model_dump(),
            "assembled_images": [],
        }

    # Build visual identity context
    primary_style = ""
    color_palette = ""
    mood = ""
    if visual_identity:
        primary_style = visual_identity.primary_style
        color_palette = ", ".join(visual_identity.color_palette)
        mood = visual_identity.mood

    # Build multimodal message
    user_prompt = EDITORIAL_USER.format(
        n_images=len(non_header_images),
        cuts_count=cuts_count,
        primary_style=primary_style,
        color_palette=color_palette,
        mood=mood,
    )

    content_parts: list[dict] = [{"type": "text", "text": user_prompt}]

    for img in non_header_images:
        image_bytes = img["image_bytes"]
        if not image_bytes:
            continue
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning(
                "Skipping oversized image '%s' (%d bytes, limit %d) from editorial review",
                img["location_id"],
                len(image_bytes),
                MAX_IMAGE_SIZE,
            )
            continue
        media_type = detect_media_type(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content_parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )
        content_parts.append(
            {
                "type": "text",
                "text": f"Image above is '{img['location_id']}' ({img['image_type']}, {img['purpose']})",
            }
        )

    try:
        response = await invoke(
            tier=ModelTier.SONNET,
            system=EDITORIAL_SYSTEM,
            user=content_parts,
            schema=EditorialReviewResult,
            config=InvokeConfig(cache=False),
        )

        # Validate cut count doesn't exceed requested
        valid_location_ids = {img["location_id"] for img in non_header_images}
        cut_ids = list(dict.fromkeys(cid for cid in response.cut_location_ids if cid in valid_location_ids))
        cut_ids = cut_ids[:cuts_count]  # Never cut more than requested

        result = response.model_copy(update={"cut_location_ids": cut_ids})

        for cut_id in cut_ids:
            # Find cut reason from evaluations
            reason = "no reason given"
            for ev in result.evaluations:
                if ev.location_id == cut_id and ev.cut_reason:
                    reason = ev.cut_reason
                    break
            logger.info(f"Editorial cut: {cut_id} — {reason}")

        logger.info(
            f"Editorial review complete: {len(cut_ids)} cuts from "
            f"{len(non_header_images)} non-header images. {result.editorial_summary}"
        )

    except Exception as e:
        logger.warning(f"Editorial review failed, keeping all images: {e}")
        result = EditorialReviewResult(
            evaluations=[],
            cut_location_ids=[],
            editorial_summary="Editorial review failed. All images kept.",
        )

    return {
        "editorial_review_result": result.model_dump(),
        "assembled_images": [],  # Clear to free memory
    }
