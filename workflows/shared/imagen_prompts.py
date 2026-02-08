"""Structured Imagen prompt schema and builder.

Converts unstructured image briefs into Imagen-optimized prompts
with must-have elements front-loaded (per Google's Imagen prompt guide).
"""

import logging

from pydantic import BaseModel, Field

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

logger = logging.getLogger(__name__)


class ImagenPromptStructure(BaseModel):
    """LLM converts brief into Imagen-optimized prompt structure."""

    primary_subject: str = Field(
        description="The single most important element. Be very specific.",
    )
    composition: str = Field(
        description="Camera angle, framing. E.g., 'close-up', 'aerial view'",
    )
    key_elements: list[str] = Field(
        description="2-3 additional required elements, in priority order.",
    )
    style_and_mood: str = Field(
        description="Visual style, lighting, color temperature.",
    )
    context_setting: str = Field(
        description="Background and environment.",
    )


STRUCTURE_SYSTEM = """You are converting an image brief into a structured format optimized for Google's Imagen model.

RULES:
- The primary_subject MUST be the single most important visual element
- key_elements should have 2-3 items maximum, ordered by importance
- composition should use photography terminology (close-up, wide shot, etc.)
- style_and_mood should specify lighting, color palette, and visual style
- context_setting describes the background/environment
- Be SPECIFIC and CONCRETE — avoid abstract descriptions
- Use positive framing — describe what to include, not what to avoid"""

STRUCTURE_USER = """Convert this image brief into a structured Imagen prompt:

{brief}"""


def build_imagen_prompt(structure: ImagenPromptStructure) -> str:
    """Build prompt with must-have elements front-loaded.

    Imagen pays most attention to the beginning of the prompt,
    so primary subject and key elements come first.
    """
    parts = [
        structure.primary_subject,
        ", ".join(structure.key_elements),
        structure.composition,
        structure.context_setting,
        structure.style_and_mood,
    ]
    return ", ".join(p for p in parts if p)


async def structure_brief_for_imagen(brief: str) -> str:
    """Convert an unstructured brief into an Imagen-optimized prompt.

    Uses Haiku to convert the brief into an ImagenPromptStructure,
    then builds a properly ordered prompt string.

    Args:
        brief: Unstructured image brief/description

    Returns:
        Structured Imagen prompt string
    """
    try:
        structure = await invoke(
            tier=ModelTier.HAIKU,
            system=STRUCTURE_SYSTEM,
            user=STRUCTURE_USER.format(brief=brief),
            schema=ImagenPromptStructure,
            config=InvokeConfig(
                max_tokens=500,
                batch_policy=BatchPolicy.PREFER_SPEED,
            ),
        )

        prompt = build_imagen_prompt(structure)
        logger.info(f"Structured Imagen prompt: {prompt[:100]}...")
        return prompt

    except Exception as e:
        logger.warning(f"Brief structuring failed, using raw brief: {e}")
        return brief
