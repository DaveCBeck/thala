"""Pass 1: Establish visual identity and identify image opportunities."""

import logging
import re

from langsmith import traceable

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from ..config import IllustrateConfig
from ..prompts import (
    CREATIVE_DIRECTION_SYSTEM,
    CREATIVE_DIRECTION_USER,
    resolve_palette_hex,
)
from ..schemas import CreativeDirectionResult
from ..state import IllustrateState

logger = logging.getLogger(__name__)


def _extract_title_from_markdown(content: str) -> str:
    """Extract title from first H1 heading or first line."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("---"):
            return line[:100]

    return "Untitled Document"


@traceable(run_type="chain", name="Illustrate_CreativeDirection")
async def creative_direction_node(state: IllustrateState) -> dict:
    """Pass 1: Establish visual identity and identify image opportunities.

    LLM sees full document, produces:
    - VisualIdentity (palette, mood, style, avoid-list)
    - Image opportunity map (N+2 locations with rationale)
    - Editorial notes (tone, pacing)

    Returns:
        State updates with extracted_title, visual_identity,
        image_opportunities, editorial_notes.
    """
    input_data = state["input"]
    config = state.get("config") or IllustrateConfig()
    document = input_data["markdown_document"]

    title = input_data.get("title") or _extract_title_from_markdown(document)

    target_image_count = (1 if config.generate_header_image else 0) + config.additional_image_count
    extra_opportunity_count = target_image_count + 2

    try:
        result = await invoke(
            tier=ModelTier.SONNET,
            system=CREATIVE_DIRECTION_SYSTEM,
            user=CREATIVE_DIRECTION_USER.format(
                title=title,
                document=document,
                target_image_count=target_image_count,
                extra_opportunity_count=extra_opportunity_count,
                generate_header=config.generate_header_image,
            ),
            schema=CreativeDirectionResult,
            config=InvokeConfig(max_tokens=4000, batch_policy=BatchPolicy.PREFER_BALANCE),
        )

        # Resolve hex colors for diagram injection
        vi = result.visual_identity
        vi.palette_hex = resolve_palette_hex(vi.color_palette)

        logger.info(
            f"Creative direction complete: style='{vi.primary_style}', "
            f"{len(result.image_opportunities)} opportunities identified"
        )

        return {
            "extracted_title": result.document_title,
            "visual_identity": vi,
            "image_opportunities": result.image_opportunities,
            "editorial_notes": result.editorial_notes,
        }

    except Exception as e:
        logger.error(f"Creative direction failed: {e}")
        return {
            "extracted_title": title,
            "image_plan": [],
            "errors": [
                {
                    "location_id": None,
                    "severity": "error",
                    "message": "Creative direction analysis failed",
                    "stage": "analysis",
                }
            ],
            "status": "failed",
        }
