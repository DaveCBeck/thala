"""Vision review of generated images with retry logic."""

import base64
import logging

from workflows.shared.llm_utils import ModelTier, get_structured_output

from ..prompts import VISION_REVIEW_SYSTEM, VISION_REVIEW_USER
from ..schemas import ImageLocationPlan, VisionReviewResult
from ..state import ImageGenResult, ImageReviewResult, WorkflowError

logger = logging.getLogger(__name__)


async def review_image_node(state: dict) -> dict:
    """Vision review of a generated image.

    Uses Sonnet with vision to evaluate:
    - Does the image fit the document context?
    - Are there substantive errors (for diagrams: incorrect info)?
    - Are there minor issues (formatting, style)?

    Recommendations:
    - accept: Image is good
    - accept_with_warning: Minor issues, log warning but use
    - retry: Substantive problems, try again with improved brief
    - fail: Fundamental problems, skip this image

    Args:
        state: Contains generation_result, location plan, document_context

    Returns:
        State update with review_results and potentially pending_retries
    """
    gen_result: ImageGenResult = state["generation_result"]
    plan: ImageLocationPlan = state["location"]
    document_context: str = state["document_context"]

    location_id = gen_result["location_id"]

    # Skip review for failed generations
    if not gen_result["success"] or not gen_result["image_bytes"]:
        logger.debug(f"Skipping review for failed generation: {location_id}")
        return {
            "review_results": [
                ImageReviewResult(
                    location_id=location_id,
                    passed=False,
                    severity="substantive",
                    issues=["Generation failed"],
                    improved_brief=None,
                )
            ]
        }

    try:
        # Build vision request
        image_bytes = gen_result["image_bytes"]
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine media type
        media_type = "image/png"  # Default for diagrams and Imagen
        if image_bytes[:2] == b"\xff\xd8":
            media_type = "image/jpeg"

        content = [
            {
                "type": "text",
                "text": VISION_REVIEW_USER.format(
                    context=document_context[:3000],
                    purpose=plan.purpose,
                    image_type=gen_result["image_type"],
                    brief=plan.brief[:1500],
                ),
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_image,
                },
            },
        ]

        review = await get_structured_output(
            output_schema=VisionReviewResult,
            user_prompt=content,
            system_prompt=VISION_REVIEW_SYSTEM,
            tier=ModelTier.SONNET,
            max_tokens=1000,
        )

        logger.info(
            f"Vision review for {location_id}: {review.recommendation} "
            f"(fits={review.fits_context}, issues={len(review.issues)})"
        )

        # Handle based on recommendation
        if review.recommendation == "accept":
            return {
                "review_results": [
                    ImageReviewResult(
                        location_id=location_id,
                        passed=True,
                        severity=None,
                        issues=[],
                        improved_brief=None,
                    )
                ]
            }

        elif review.recommendation == "accept_with_warning":
            logger.warning(
                f"Image {location_id} accepted with warnings: {review.issues}"
            )
            return {
                "review_results": [
                    ImageReviewResult(
                        location_id=location_id,
                        passed=True,
                        severity="minor",
                        issues=review.issues,
                        improved_brief=None,
                    )
                ],
                "errors": [
                    WorkflowError(
                        location_id=location_id,
                        severity="warning",
                        message=f"Minor issues: {', '.join(review.issues[:3])}",
                        stage="review",
                    )
                ],
            }

        elif review.recommendation == "retry":
            logger.info(f"Image {location_id} needs retry: {review.issues}")
            return {
                "review_results": [
                    ImageReviewResult(
                        location_id=location_id,
                        passed=False,
                        severity="substantive",
                        issues=review.issues,
                        improved_brief=review.improved_brief,
                    )
                ],
                "pending_retries": [location_id],
                "retry_briefs": {location_id: review.improved_brief}
                if review.improved_brief
                else {},
            }

        else:  # fail
            logger.warning(
                f"Image {location_id} failed review, skipping: {review.issues}"
            )
            return {
                "review_results": [
                    ImageReviewResult(
                        location_id=location_id,
                        passed=False,
                        severity="substantive",
                        issues=review.issues,
                        improved_brief=None,
                    )
                ],
                "errors": [
                    WorkflowError(
                        location_id=location_id,
                        severity="warning",
                        message=f"Image rejected: {', '.join(review.issues[:3])}",
                        stage="review",
                    )
                ],
            }

    except Exception as e:
        logger.error(f"Vision review failed for {location_id}: {e}")
        # On review failure, accept the image rather than fail the workflow
        return {
            "review_results": [
                ImageReviewResult(
                    location_id=location_id,
                    passed=True,  # Accept on review error
                    severity="minor",
                    issues=[f"Review failed: {e}"],
                    improved_brief=None,
                )
            ],
            "errors": [
                WorkflowError(
                    location_id=location_id,
                    severity="warning",
                    message=f"Vision review failed, accepting image: {e}",
                    stage="review",
                )
            ],
        }
