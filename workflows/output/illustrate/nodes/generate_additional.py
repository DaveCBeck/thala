"""Generate additional images (public domain, diagrams, or generated)."""

import ipaddress
import logging
from urllib.parse import urlparse

import httpx

from core.images import NoResultsError, get_image
from workflows.shared.diagram_utils import DiagramConfig, generate_diagram
from workflows.shared.diagram_utils.registry import is_engine_available
from workflows.shared.image_utils import generate_article_header

from ..config import IllustrateConfig
from ..prompts import build_visual_identity_context
from ..schemas import ImageLocationPlan, VisualIdentity
from ..state import ImageGenResult, WorkflowError

logger = logging.getLogger(__name__)

# Diagram subtypes grouped by preferred rendering engine
_MERMAID_SUBTYPES = {"flowchart", "sequence", "concept_map"}
_GRAPHVIZ_SUBTYPES = {"network_graph", "hierarchy", "dependency_tree"}


def _validate_image_url(url: str) -> None:
    """Validate URL is external HTTPS to prevent SSRF."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs allowed, got: {parsed.scheme}")
    hostname = parsed.hostname or ""
    # Block internal/loopback addresses
    if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise ValueError(f"Internal addresses not allowed: {hostname}")
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_reserved:
            raise ValueError(f"Private/reserved IP not allowed: {hostname}")
    except ValueError as exc:
        # Re-raise our own validation errors; ignore errors from
        # ip_address() when hostname is a regular domain name.
        if "not allowed" in str(exc):
            raise
        # hostname is not an IP literal — that's fine (it's a domain name)


async def _download_image(url: str) -> bytes:
    """Download image from URL."""
    _validate_image_url(url)
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
    visual_identity: VisualIdentity | None = state.get("visual_identity")
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
                visual_identity=visual_identity,
            )

        elif image_type == "generated":
            return await _generate_imagen(
                location_id=location_id,
                plan=plan,
                brief=brief,
                config=config,
                visual_identity=visual_identity,
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
    """Generate using public domain image search.

    Uses multi-query search_pool() when literal/conceptual queries are
    available, falling back to single-query get_image() otherwise.
    """
    try:
        # Build query list from multi-query fields if available
        queries = _build_search_queries(plan)

        if queries:
            # Multi-query path: search_pool + LLM selection
            from core.images.service import get_image_service
            from core.images.selection import select_best_image

            service = get_image_service()
            pool = await service.search_pool(
                queries=queries,
                limit_per_query=3,
                orientation="landscape",
            )

            if not pool:
                raise NoResultsError(
                    "No images found for multi-query search",
                    provider="all",
                )

            result = await select_best_image(
                pool,
                query=queries[0],
                context=document_context,
                custom_selection_criteria=brief,
            )
        else:
            # Fallback: single-query via get_image()
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
                    prompt_or_query_used=str(queries or plan.search_query or brief[:100]),
                    alt_text=result.metadata.alt_text or result.metadata.description,
                    attribution=result.attribution.model_dump() if result.attribution else None,
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


def _build_search_queries(plan: ImageLocationPlan) -> list[str]:
    """Build prioritized query list from plan's multi-query fields.

    Returns empty list if no multi-query fields are populated,
    signaling the caller to fall back to single-query search.
    """
    if not plan.literal_queries and not plan.conceptual_queries:
        return []

    strategy = plan.query_strategy or "both"

    if strategy == "literal":
        return plan.literal_queries[:4]
    elif strategy == "conceptual":
        return plan.conceptual_queries[:4]
    else:  # "both"
        # Interleave conceptual and literal for diversity
        queries: list[str] = []
        for conceptual, literal in zip(plan.conceptual_queries, plan.literal_queries):
            queries.extend([conceptual, literal])
        # Add any remaining
        queries.extend(plan.conceptual_queries[len(plan.literal_queries) :])
        queries.extend(plan.literal_queries[len(plan.conceptual_queries) :])
        return queries[:4]


async def _generate_diagram(
    location_id: str,
    plan: ImageLocationPlan,
    brief: str,
    config: IllustrateConfig,
    visual_identity: VisualIdentity | None = None,
) -> dict:
    """Generate diagram, routing to the best engine based on subtype.

    Routing:
    - flowchart/sequence/concept_map → Mermaid (with vision selection)
    - network_graph/hierarchy/dependency_tree → Graphviz (with vision selection)
    - custom_artistic/None → raw SVG (existing pipeline)

    Falls back to SVG if the preferred engine is unavailable or fails.
    """
    diagram_config = DiagramConfig(
        width=config.diagram_width,
        height=config.diagram_height,
        enable_refinement_loop=config.enable_diagram_refinement,
        quality_threshold=config.diagram_quality_threshold,
        max_refinement_iterations=config.diagram_max_refinement_iterations,
    )

    # Inject visual identity context (includes avoid list for LLM-consumed prompts)
    vi_context = build_visual_identity_context(visual_identity)
    diagram_brief = brief + vi_context if vi_context else brief

    subtype = plan.diagram_subtype
    result = None

    # Try preferred engine based on subtype
    if subtype in _MERMAID_SUBTYPES and is_engine_available("mermaid"):
        from workflows.shared.diagram_utils.mermaid import generate_mermaid_with_selection

        logger.info(f"Routing diagram {location_id} to Mermaid engine (subtype={subtype})")
        result = await generate_mermaid_with_selection(
            analysis=diagram_brief,
            config=diagram_config,
            custom_instructions=diagram_brief,
        )

    elif subtype in _GRAPHVIZ_SUBTYPES and is_engine_available("graphviz"):
        from workflows.shared.diagram_utils.graphviz_engine import generate_graphviz_with_selection

        logger.info(f"Routing diagram {location_id} to Graphviz engine (subtype={subtype})")
        result = await generate_graphviz_with_selection(
            analysis=diagram_brief,
            config=diagram_config,
            custom_instructions=diagram_brief,
        )

    # Fallback to SVG if preferred engine failed or wasn't available
    if result is None or not result.success:
        if result and not result.success:
            logger.warning(
                f"Preferred engine failed for {location_id} (subtype={subtype}): {result.error}, falling back to SVG"
            )
        result = await generate_diagram(
            title="",
            content="",
            config=diagram_config,
            custom_instructions=diagram_brief,
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
    visual_identity: VisualIdentity | None = None,
) -> dict:
    """Generate using Imagen."""
    # Inject visual identity (for_imagen=True omits avoid list)
    vi_context = build_visual_identity_context(visual_identity, for_imagen=True)
    imagen_brief = brief + vi_context if vi_context else brief

    image_bytes, prompt_used = await generate_article_header(
        title="",
        content="",
        custom_prompt=imagen_brief,
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
