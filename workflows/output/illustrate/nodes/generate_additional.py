"""Image generation helpers (public domain, diagrams, Imagen).

Used by generate_candidate.py via the _generate_* helper functions.
"""

import ipaddress
import logging
from urllib.parse import urlparse

import httpx

from core.images import NoResultsError, get_image
from workflows.shared.image_utils import generate_article_header, generate_diagram_image

from ..config import IllustrateConfig
from ..prompts import build_visual_identity_context
from ..schemas import ImageLocationPlan, VisualIdentity
from ..state import ImageGenResult, WorkflowError

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB


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
    """Download image from URL with size limit to prevent memory exhaustion."""
    _validate_image_url(url)
    async with httpx.AsyncClient(timeout=30.0) as client:
        chunks: list[bytes] = []
        total = 0
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image exceeds size limit: >{MAX_IMAGE_SIZE} bytes ({MAX_IMAGE_SIZE // (1024 * 1024)} MB)"
                    )
                chunks.append(chunk)
        return b"".join(chunks)


async def _generate_public_domain(
    location_id: str,
    plan: ImageLocationPlan,
    brief: str,
    document_context: str,
    brief_id: str = "",
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
                    brief_id=brief_id,
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
                    brief_id=brief_id,
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
    brief_id: str = "",
) -> dict:
    """Generate diagram using Gemini 3 Pro image generation.

    Uses for_imagen=True for visual identity injection (same "avoid" paradox
    as Imagen — embedding "avoid X" in positive prompts causes generation of X).

    Note: custom_artistic is intercepted by generate_candidate_node and
    routed to Imagen instead — it never reaches this function.
    """
    vi_context = build_visual_identity_context(visual_identity, for_imagen=True)
    diagram_brief = brief + vi_context if vi_context else brief

    image_bytes, prompt_used = await generate_diagram_image(brief=diagram_brief)

    if image_bytes:
        logger.info(f"Generated diagram for {location_id}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    brief_id=brief_id,
                    success=True,
                    image_bytes=image_bytes,
                    image_type="diagram",
                    prompt_or_query_used=prompt_used or brief[:200],
                    alt_text=f"Diagram: {plan.insertion_after_header}",
                    attribution=None,
                )
            ]
        }
    else:
        logger.error(f"Diagram generation failed for {location_id}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    brief_id=brief_id,
                    success=False,
                    image_bytes=None,
                    image_type="diagram",
                    prompt_or_query_used=prompt_used or brief[:200],
                    alt_text=None,
                    attribution=None,
                )
            ],
            "errors": [
                WorkflowError(
                    location_id=location_id,
                    severity="error",
                    message="Diagram generation failed: Gemini returned no image",
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
    brief_id: str = "",
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
        sample_count=config.imagen_sample_count,
    )

    if image_bytes:
        logger.info(f"Generated Imagen image for {location_id}")
        return {
            "generation_results": [
                ImageGenResult(
                    location_id=location_id,
                    brief_id=brief_id,
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
                    brief_id=brief_id,
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
