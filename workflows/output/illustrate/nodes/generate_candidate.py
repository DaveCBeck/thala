"""Generate a single image candidate from a CandidateBrief."""

import logging

from ..config import IllustrateConfig
from ..schemas import CandidateBrief, ImageLocationPlan, VisualIdentity
from ..state import ImageGenResult, WorkflowError
from .generate_additional import (
    _generate_diagram,
    _generate_imagen,
    _generate_public_domain,
)

logger = logging.getLogger(__name__)


async def generate_candidate_node(state: dict) -> dict:
    """Generate a single image candidate from a CandidateBrief.

    Receives brief + location plan, routes to the appropriate generator,
    and tags the result with brief_id for downstream grouping.

    Args:
        state: Contains location, brief, brief_id, document_context, config, visual_identity

    Returns:
        State update with generation_results (tagged with brief_id)
    """
    plan: ImageLocationPlan = state["location"]
    brief: CandidateBrief = state["brief"]
    brief_id: str = state["brief_id"]
    document_context: str = state["document_context"]
    config: IllustrateConfig = state.get("config") or IllustrateConfig()
    visual_identity: VisualIdentity | None = state.get("visual_identity")

    location_id = plan.location_id
    image_type = brief.image_type
    brief_text = brief.brief

    # custom_artistic subtype: step 2 overriding step 1's "diagram" suggestion
    # to request an artistic/painterly image instead — route to Imagen.
    is_artistic_override = (
        image_type == "diagram" and brief.diagram_subtype == "custom_artistic"
    )

    try:
        if image_type == "public_domain":
            result = await _generate_public_domain(
                location_id=location_id,
                plan=plan,
                brief=brief_text,
                document_context=document_context,
                brief_id=brief_id,
            )
        elif image_type == "diagram" and not is_artistic_override:
            result = await _generate_diagram(
                location_id=location_id,
                plan=plan,
                brief=brief_text,
                config=config,
                visual_identity=visual_identity,
                brief_id=brief_id,
            )
        elif image_type == "generated" or is_artistic_override:
            if is_artistic_override:
                logger.info(f"Routing custom_artistic diagram {location_id} to Imagen")
            result = await _generate_imagen(
                location_id=location_id,
                plan=plan,
                brief=brief_text,
                config=config,
                visual_identity=visual_identity,
                brief_id=brief_id,
            )
        else:
            logger.error(f"Unknown image type: {image_type}")
            return _failure_result(location_id, brief_id, image_type, brief_text, f"Unknown image type: {image_type}")

        return result

    except Exception as e:
        logger.error(f"Candidate generation failed for {brief_id}: {e}")
        return _failure_result(location_id, brief_id, image_type, brief_text, f"Generation failed: {e}")


def _failure_result(
    location_id: str,
    brief_id: str,
    image_type: str,
    brief_text: str,
    message: str,
) -> dict:
    """Build a failure result dict."""
    return {
        "generation_results": [
            ImageGenResult(
                location_id=location_id,
                brief_id=brief_id,
                success=False,
                image_bytes=None,
                image_type=image_type,
                prompt_or_query_used=brief_text,
                alt_text=None,
                attribution=None,
            )
        ],
        "errors": [
            WorkflowError(
                location_id=location_id,
                severity="error",
                message=message,
                stage="generation",
            )
        ],
    }
