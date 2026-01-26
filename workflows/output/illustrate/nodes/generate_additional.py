"""Generate additional images (public domain, diagrams, or generated)."""

import logging

import httpx

from core.images import NoResultsError, get_image
from workflows.shared.diagram_utils import DiagramConfig, generate_diagram
from workflows.shared.image_utils import generate_article_header

from ..config import IllustrateConfig
from ..schemas import ImageLocationPlan
from ..state import ImageGenResult, WorkflowError

logger = logging.getLogger(__name__)


async def _download_image(url: str) -> bytes:
    """Download image from URL."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def generate_additional_node(state: dict) -> dict:
    """Generate an additional image based on the plan.

    This node is invoked via Send() for each additional image location.
    It handles all three image types:
    - public_domain: Search and download from Pexels/Unsplash
    - diagram: Generate SVG diagram and convert to PNG
    - generated: Use Imagen to generate

    Args:
        state: Contains location plan, document_context, config

    Returns:
        State update with generation_results
    """
    plan: ImageLocationPlan = state["location"]
    document_context: str = state["document_context"]
    config: IllustrateConfig = state.get("config") or IllustrateConfig()
    retry_brief: str | None = state.get("retry_brief")

    # Use retry brief if this is a retry
    brief = retry_brief or plan.brief
    location_id = plan.location_id
    image_type = plan.image_type

    try:
        if image_type == "public_domain":
            return await _generate_public_domain(
                location_id=location_id,
                plan=plan,
                brief=brief,
                document_context=document_context,
            )

        elif image_type == "diagram":
            return await _generate_diagram(
                location_id=location_id,
                plan=plan,
                brief=brief,
                config=config,
            )

        elif image_type == "generated":
            return await _generate_imagen(
                location_id=location_id,
                plan=plan,
                brief=brief,
                config=config,
            )

        else:
            logger.error(f"Unknown image type: {image_type}")
            return {
                "generation_results": [
                    ImageGenResult(
                        location_id=location_id,
                        success=False,
                        image_bytes=None,
                        image_type=image_type,
                        prompt_or_query_used=brief,
                        alt_text=None,
                        attribution=None,
                    )
                ],
                "errors": [
                    WorkflowError(
                        location_id=location_id,
                        severity="error",
                        message=f"Unknown image type: {image_type}",
                        stage="generation",
                    )
                ],
            }

    except Exception as e:
        logger.error(f"Image generation failed for {location_id}: {e}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=False,
                    image_bytes=None,
                    image_type=image_type,
                    prompt_or_query_used=brief,
                    alt_text=None,
                    attribution=None,
                )
            ],
            "errors": [
                WorkflowError(
                    location_id=location_id,
                    severity="error",
                    message=f"Generation failed: {e}",
                    stage="generation",
                )
            ],
        }


async def _generate_public_domain(
    location_id: str,
    plan: ImageLocationPlan,
    brief: str,
    document_context: str,
) -> dict:
    """Generate using public domain image search."""
    try:
        search_query = plan.search_query or brief[:100]

        result = await get_image(
            query=search_query,
            use_llm_selection=True,
            context=document_context,
            custom_selection_criteria=brief,
            orientation="landscape",
        )

        image_bytes = await _download_image(result.url)
        logger.info(f"Downloaded public domain image for {location_id}")

        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=True,
                    image_bytes=image_bytes,
                    image_type="public_domain",
                    prompt_or_query_used=search_query,
                    alt_text=result.metadata.alt_text or result.metadata.description,
                    attribution=result.attribution.model_dump()
                    if result.attribution
                    else None,
                )
            ]
        }

    except NoResultsError as e:
        logger.warning(f"No public domain results for {location_id}: {e}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=False,
                    image_bytes=None,
                    image_type="public_domain",
                    prompt_or_query_used=brief[:100],
                    alt_text=None,
                    attribution=None,
                )
            ],
            "errors": [
                WorkflowError(
                    location_id=location_id,
                    severity="warning",
                    message=f"No public domain images found: {e}",
                    stage="generation",
                )
            ],
        }


async def _generate_diagram(
    location_id: str,
    plan: ImageLocationPlan,
    brief: str,
    config: IllustrateConfig,
) -> dict:
    """Generate diagram using diagram_utils."""
    diagram_config = DiagramConfig(
        width=config.diagram_width,
        height=config.diagram_height,
        enable_refinement_loop=config.enable_diagram_refinement,
        quality_threshold=config.diagram_quality_threshold,
        max_refinement_iterations=config.diagram_max_refinement_iterations,
    )

    result = await generate_diagram(
        title="",
        content="",
        config=diagram_config,
        custom_instructions=brief,
    )

    if result.success and result.png_bytes:
        logger.info(f"Generated diagram for {location_id}")

        # Check for overlap warnings
        errors = []
        if result.error:  # Contains overlap info even on success
            errors.append(
                WorkflowError(
                    location_id=location_id,
                    severity="warning",
                    message=result.error,
                    stage="generation",
                )
            )

        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=True,
                    image_bytes=result.png_bytes,
                    image_type="diagram",
                    prompt_or_query_used=brief[:200],
                    alt_text=f"Diagram: {plan.insertion_after_header}",
                    attribution=None,
                )
            ],
            "errors": errors if errors else [],
        }
    else:
        logger.error(f"Diagram generation failed for {location_id}: {result.error}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=False,
                    image_bytes=None,
                    image_type="diagram",
                    prompt_or_query_used=brief[:200],
                    alt_text=None,
                    attribution=None,
                )
            ],
            "errors": [
                WorkflowError(
                    location_id=location_id,
                    severity="error",
                    message=f"Diagram generation failed: {result.error}",
                    stage="generation",
                )
            ],
        }


async def _generate_imagen(
    location_id: str,
    plan: ImageLocationPlan,
    brief: str,
    config: IllustrateConfig,
) -> dict:
    """Generate using Imagen."""
    image_bytes, prompt_used = await generate_article_header(
        title="",
        content="",
        custom_prompt=brief,
        aspect_ratio=config.imagen_aspect_ratio,
    )

    if image_bytes:
        logger.info(f"Generated Imagen image for {location_id}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    success=True,
                    image_bytes=image_bytes,
                    image_type="generated",
                    prompt_or_query_used=prompt_used or brief,
                    alt_text=f"Illustration for {plan.insertion_after_header}",
                    attribution=None,
                )
            ]
        }
    else:
        logger.error(f"Imagen returned no image for {location_id}")
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
            "errors": [
                WorkflowError(
                    location_id=location_id,
                    severity="error",
                    message="Imagen generation returned no image",
                    stage="generation",
                )
            ],
        }
