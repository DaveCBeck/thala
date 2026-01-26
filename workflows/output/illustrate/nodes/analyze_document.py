"""Analyze document and plan image locations."""

import logging
import re

from workflows.shared.llm_utils import ModelTier, get_structured_output

from ..config import IllustrateConfig
from ..prompts import ANALYSIS_SYSTEM, ANALYSIS_USER
from ..schemas import DocumentAnalysis
from ..state import IllustrateState

logger = logging.getLogger(__name__)


def _extract_title_from_markdown(content: str) -> str:
    """Extract title from first H1 heading or first line."""
    # Look for # Title
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fall back to first non-empty line
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("---"):
            return line[:100]

    return "Untitled Document"


async def analyze_document_node(state: IllustrateState) -> dict:
    """Analyze document and plan image locations.

    Uses Sonnet structured output to determine:
    - Where images should go (after which headers)
    - What type each should be (generated/public_domain/diagram)
    - Detailed briefs for each image

    Returns:
        State updates with extracted_title and image_plan
    """
    input_data = state["input"]
    config = state.get("config") or IllustrateConfig()
    document = input_data["markdown_document"]

    # Extract or use provided title
    title = input_data.get("title") or _extract_title_from_markdown(document)

    try:
        analysis = await get_structured_output(
            output_schema=DocumentAnalysis,
            user_prompt=ANALYSIS_USER.format(
                title=title,
                document=document[:12000],  # Truncate for token limits
                generate_header=config.generate_header_image,
                additional_count=config.additional_image_count,
                prefer_pd_header=config.header_prefer_public_domain,
            ),
            system_prompt=ANALYSIS_SYSTEM,
            tier=ModelTier.SONNET,
            max_tokens=4000,
        )

        # Build image plan list
        image_plan = []

        if config.generate_header_image:
            # For header, we always try public_domain first per user requirement
            header_plan = analysis.header_image
            if config.header_prefer_public_domain:
                header_plan.image_type = "public_domain"
            image_plan.append(header_plan)

        # Add additional images up to configured count
        for plan in analysis.additional_images[: config.additional_image_count]:
            image_plan.append(plan)

        logger.info(
            f"Document analysis complete: {len(image_plan)} images planned "
            f"({analysis.analysis_notes[:100]}...)"
        )

        return {
            "extracted_title": analysis.document_title,
            "image_plan": image_plan,
        }

    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        return {
            "extracted_title": title,
            "image_plan": [],
            "errors": [
                {
                    "location_id": None,
                    "severity": "error",
                    "message": f"Document analysis failed: {e}",
                    "stage": "analysis",
                }
            ],
            "status": "failed",
        }
