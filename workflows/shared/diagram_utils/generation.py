"""LLM-based diagram analysis and SVG generation.

Contains functions for analyzing content to determine diagram requirements
and generating SVG diagrams using Claude.
"""

import logging

from ..llm_utils import ModelTier, get_structured_output
from .prompts import (
    DIAGRAM_ANALYSIS_SYSTEM,
    DIAGRAM_ANALYSIS_USER,
    SVG_GENERATION_SYSTEM,
    SVG_GENERATION_USER,
    SVG_REGENERATION_USER,
)
from .schemas import DiagramAnalysis, DiagramConfig, OverlapCheckResult

logger = logging.getLogger(__name__)


async def analyze_content_for_diagram(
    title: str,
    content: str,
    tier: ModelTier = ModelTier.HAIKU,
) -> DiagramAnalysis | None:
    """Analyze content to determine if/how to create a diagram.

    Uses structured output to get a reliable DiagramAnalysis schema.

    Args:
        title: Content title
        content: Full content text (will be truncated to ~8000 chars)
        tier: Model tier for analysis (default HAIKU for speed/cost)

    Returns:
        DiagramAnalysis if successful, None on failure
    """
    # Truncate content if too long
    truncated_content = content[:8000] if len(content) > 8000 else content

    try:
        result = await get_structured_output(
            output_schema=DiagramAnalysis,
            user_prompt=DIAGRAM_ANALYSIS_USER.format(
                title=title, content=truncated_content
            ),
            system_prompt=DIAGRAM_ANALYSIS_SYSTEM,
            tier=tier,
            max_tokens=1000,
        )
        logger.info(
            f"Diagram analysis for '{title}': "
            f"should_generate={result.should_generate}, type={result.diagram_type}"
        )
        return result
    except Exception as e:
        logger.error(f"Failed to analyze content for diagram: {e}")
        return None


async def generate_svg_diagram(
    analysis: DiagramAnalysis,
    config: DiagramConfig,
    tier: ModelTier = ModelTier.SONNET,
) -> str | None:
    """Generate SVG code from diagram analysis.

    Args:
        analysis: DiagramAnalysis from analyze_content_for_diagram
        config: Diagram configuration
        tier: Model tier (default SONNET for quality SVG generation)

    Returns:
        SVG code string if successful, None on failure
    """
    from langchain_anthropic import ChatAnthropic

    try:
        llm = ChatAnthropic(
            model=tier.value,
            max_tokens=4000,
        )

        # Format the system prompt with dimensions
        system_prompt = SVG_GENERATION_SYSTEM.format(
            width=config.width,
            height=config.height,
            background=config.background_color,
        )

        user_prompt = SVG_GENERATION_USER.format(
            diagram_type=analysis.diagram_type.value,
            title=analysis.title,
            elements=", ".join(analysis.key_elements),
            relationships="; ".join(analysis.relationships),
            width=config.width,
            height=config.height,
            primary_color=config.primary_color,
            background_color=config.background_color,
            font_family=config.font_family,
        )

        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        svg_content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()

        # Clean up response - remove any markdown fences
        if svg_content.startswith("```"):
            lines = svg_content.split("\n")
            # Remove first line (```svg or ```) and last line (```)
            svg_content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Ensure it starts with <svg
        if not svg_content.strip().startswith("<svg"):
            logger.warning("Generated content doesn't start with <svg>")
            # Try to extract SVG from response
            start = svg_content.find("<svg")
            end = svg_content.rfind("</svg>")
            if start != -1 and end != -1:
                svg_content = svg_content[start : end + 6]
            else:
                logger.error("Could not extract valid SVG from response")
                return None

        logger.info(f"Generated SVG diagram ({len(svg_content)} chars)")
        return svg_content

    except Exception as e:
        logger.error(f"Failed to generate SVG diagram: {e}")
        return None


async def regenerate_svg_with_feedback(
    analysis: DiagramAnalysis,
    config: DiagramConfig,
    overlap_check: OverlapCheckResult,
    tier: ModelTier = ModelTier.SONNET,
) -> str | None:
    """Regenerate SVG with feedback about overlap issues."""
    from langchain_anthropic import ChatAnthropic

    try:
        llm = ChatAnthropic(
            model=tier.value,
            max_tokens=4000,
        )

        # Format overlap issues for feedback
        overlap_issues = "\n".join(
            [f'- "{t1}" overlaps with "{t2}"' for t1, t2 in overlap_check.overlap_pairs]
        )

        system_prompt = SVG_GENERATION_SYSTEM.format(
            width=config.width,
            height=config.height,
            background=config.background_color,
        )

        user_prompt = SVG_REGENERATION_USER.format(
            diagram_type=analysis.diagram_type.value,
            title=analysis.title,
            elements=", ".join(analysis.key_elements),
            relationships="; ".join(analysis.relationships),
            overlap_issues=overlap_issues,
            width=config.width,
            height=config.height,
            primary_color=config.primary_color,
            background_color=config.background_color,
            font_family=config.font_family,
        )

        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        svg_content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()

        # Clean up response
        if svg_content.startswith("```"):
            lines = svg_content.split("\n")
            svg_content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        if not svg_content.strip().startswith("<svg"):
            start = svg_content.find("<svg")
            end = svg_content.rfind("</svg>")
            if start != -1 and end != -1:
                svg_content = svg_content[start : end + 6]
            else:
                return None

        logger.info(f"Regenerated SVG diagram ({len(svg_content)} chars)")
        return svg_content

    except Exception as e:
        logger.error(f"Failed to regenerate SVG diagram: {e}")
        return None


__all__ = [
    "analyze_content_for_diagram",
    "generate_svg_diagram",
    "regenerate_svg_with_feedback",
]
