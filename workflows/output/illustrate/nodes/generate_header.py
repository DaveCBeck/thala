"""Generate header image with public domain first, fallback to Imagen."""

import base64
import logging

import httpx

from core.images import NoResultsError, get_image
from workflows.shared.image_utils import generate_article_header
from workflows.shared.llm_utils import ModelTier, invoke, InvokeConfig

from ..config import IllustrateConfig
from ..prompts import HEADER_APPOSITES_SYSTEM, HEADER_APPOSITES_USER
from ..schemas import HeaderAppositenessResult, ImageLocationPlan
from ..state import ImageGenResult, WorkflowError

logger = logging.getLogger(__name__)


async def _download_image(url: str) -> bytes:
    """Download image from URL."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def _evaluate_pd_appositeness(
    image_bytes: bytes,
    document_context: str,
    query: str,
    criteria: str,
) -> tuple[bool, str, str | None]:
    """Use vision to evaluate if public domain image is 'apposite'.

    Returns:
        Tuple of (is_apposite, reasoning, suggested_search_query)
    """
    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine media type (assume JPEG for most stock photos)
        media_type = "image/jpeg"
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            media_type = "image/png"

        content = [
            {
                "type": "text",
                "text": HEADER_APPOSITES_USER.format(
                    context=document_context[:3000],
                    query=query,
                    criteria=criteria[:1000],
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

        result = await invoke(
            tier=ModelTier.SONNET,
            system=HEADER_APPOSITES_SYSTEM,
            user=content,
            schema=HeaderAppositenessResult,
            config=InvokeConfig(max_tokens=500),
        )

        # Score 3+ means apposite
        is_apposite = result.is_apposite and result.quality_score >= 3
        logger.info(
            f"Header PD appositeness: {is_apposite} (score={result.quality_score}, reason={result.reasoning[:100]})"
        )
        return is_apposite, result.reasoning, result.suggested_search_query

    except Exception as e:
        logger.warning(f"Appositeness evaluation failed, falling back to Imagen: {e}")
        return False, str(e), None


async def generate_header_node(state: dict) -> dict:
    """Generate header image: try public domain first, fallback to Imagen.

    This node handles the special header flow:
    1. Search public domain images
    2. Use vision to evaluate if result is 'apposite'
    3. If apposite, use the PD image
    4. If not, generate with Imagen

    Args:
        state: Contains location plan, document_context, config

    Returns:
        State update with generation_results
    """
    plan: ImageLocationPlan = state["location"]
    document_context: str = state["document_context"]
    config: IllustrateConfig = state.get("config") or IllustrateConfig()
    is_retry: bool = state.get("is_retry", False)
    retry_brief: str | None = state.get("retry_brief")

    # Use retry brief if this is a retry
    brief = retry_brief or plan.brief
    location_id = plan.location_id

    errors: list[WorkflowError] = []

    # Step 1: Try public domain (unless this is a retry or PD not preferred)
    # We'll try up to 2 PD searches before falling back to Imagen
    max_pd_attempts = 2
    pd_attempt = 0
    current_search_query = plan.search_query or brief[:100]

    if config.header_prefer_public_domain and not is_retry:
        while pd_attempt < max_pd_attempts:
            pd_attempt += 1
            try:
                pd_result = await get_image(
                    query=current_search_query,
                    use_llm_selection=True,
                    context=document_context,
                    custom_selection_criteria=brief,
                    orientation="landscape",
                )

                # Download and evaluate
                image_bytes = await _download_image(pd_result.url)
                is_apposite, reasoning, suggested_query = await _evaluate_pd_appositeness(
                    image_bytes=image_bytes,
                    document_context=document_context,
                    query=current_search_query,
                    criteria=brief,
                )

                if is_apposite:
                    logger.info("Using apposite public domain image for header")
                    return {
                        "generation_results": [
                            ImageGenResult(
                                location_id=location_id,
                                success=True,
                                image_bytes=image_bytes,
                                image_type="public_domain",
                                prompt_or_query_used=current_search_query,
                                alt_text=pd_result.metadata.alt_text or pd_result.metadata.description,
                                attribution=pd_result.attribution.model_dump() if pd_result.attribution else None,
                            )
                        ]
                    }
                else:
                    # If we have a suggested query and more attempts, retry with it
                    if suggested_query and pd_attempt < max_pd_attempts:
                        logger.info(
                            f"PD image not apposite, retrying with suggested query: {suggested_query}"
                        )
                        current_search_query = suggested_query
                    else:
                        logger.info(f"PD image not apposite ({reasoning}), falling back to Imagen")
                        break

            except NoResultsError:
                if pd_attempt < max_pd_attempts:
                    logger.info(f"No PD results for '{current_search_query}', will try Imagen")
                else:
                    logger.info("No public domain results, falling back to Imagen")
                break
            except Exception as e:
                logger.warning(f"Public domain search failed: {e}, falling back to Imagen")
                errors.append(
                    WorkflowError(
                        location_id=location_id,
                        severity="warning",
                        message=f"PD search failed, using Imagen: {e}",
                        stage="generation",
                    )
                )
                break

    # Step 2: Generate with Imagen
    try:
        image_bytes, prompt_used = await generate_article_header(
            title="",
            content="",
            custom_prompt=brief,
            aspect_ratio=config.imagen_aspect_ratio,
        )

        if image_bytes:
            logger.info(f"Generated header image with Imagen ({len(image_bytes)} bytes)")
            return {
                "generation_results": [
                    ImageGenResult(
                        location_id=location_id,
                        success=True,
                        image_bytes=image_bytes,
                        image_type="generated",
                        prompt_or_query_used=prompt_used or brief,
                        alt_text=f"Header image for {plan.insertion_after_header}",
                        attribution=None,
                    )
                ],
                "errors": errors if errors else [],
            }
        else:
            logger.error("Imagen returned no image")
            return {
                "generation_results": [
                    ImageGenResult(
                        location_id=location_id,
                        success=False,
                        image_bytes=None,
                        image_type="generated",
                        prompt_or_query_used=brief,
                        alt_text=None,
                        attribution=None,
                    )
                ],
                "errors": errors
                + [
                    WorkflowError(
                        location_id=location_id,
                        severity="error",
                        message="Imagen generation returned no image",
                        stage="generation",
                    )
                ],
            }

    except Exception as e:
        logger.error(f"Header image generation failed: {e}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=False,
                    image_bytes=None,
                    image_type="generated",
                    prompt_or_query_used=brief,
                    alt_text=None,
                    attribution=None,
                )
            ],
            "errors": errors
            + [
                WorkflowError(
                    location_id=location_id,
                    severity="error",
                    message=f"Header generation failed: {e}",
                    stage="generation",
                )
            ],
        }
