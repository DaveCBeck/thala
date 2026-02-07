"""LLM-based diagram analysis and SVG generation.

Contains functions for analyzing content to determine diagram requirements
and generating SVG diagrams using Claude.
"""

import logging

from ..llm_utils import invoke, InvokeConfig, ModelTier
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
    tier: ModelTier = ModelTier.OPUS,
) -> DiagramAnalysis | None:
    """Analyze content to determine if/how to create a diagram.

    Uses structured output to get a reliable DiagramAnalysis schema.

    Args:
        title: Content title
        content: Full content text (will be truncated to ~8000 chars)
        tier: Model tier for analysis (default OPUS for speed/cost)

    Returns:
        DiagramAnalysis if successful, None on failure
    """
    # Truncate content if too long
    truncated_content = content[:8000] if len(content) > 8000 else content

    try:
        result = await invoke(
            tier=tier,
            system=DIAGRAM_ANALYSIS_SYSTEM,
            user=DIAGRAM_ANALYSIS_USER.format(
                title=title, content=truncated_content
            ),
            schema=DiagramAnalysis,
            config=InvokeConfig(max_tokens=1000),
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
    tier: ModelTier = ModelTier.OPUS,
) -> str | None:
    """Generate SVG code from diagram analysis.

    Args:
        analysis: DiagramAnalysis from analyze_content_for_diagram
        config: Diagram configuration
        tier: Model tier (default OPUS for quality SVG generation)

    Returns:
        SVG code string if successful, None on failure
    """
    try:
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

        response = await invoke(
            tier=tier,
            system=system_prompt,
            user=user_prompt,
            config=InvokeConfig(max_tokens=4000),
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
    tier: ModelTier = ModelTier.OPUS,
) -> str | None:
    """Regenerate SVG with feedback about overlap issues."""
    try:
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

        response = await invoke(
            tier=tier,
            system=system_prompt,
            user=user_prompt,
            config=InvokeConfig(max_tokens=4000),
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


INSTRUCTIONS_PARSE_SYSTEM = """You are an expert at interpreting diagram specifications.
Given detailed instructions for a diagram, extract the structured components needed for SVG generation.

Determine the most appropriate diagram type from: flowchart, concept_map, process_diagram, hierarchy, comparison, timeline, cycle.

Extract the key elements (concepts, entities, steps) that should appear in the diagram.
IMPORTANT: Select at most 15 key elements (typically 5-10). Prioritize the most important concepts.
Extract all relationships or flows between elements."""


async def parse_instructions_to_analysis(
    instructions: str,
    tier: ModelTier = ModelTier.SONNET,
) -> DiagramAnalysis | None:
    """Parse custom instructions into a DiagramAnalysis schema.

    Used when the illustrate workflow provides detailed diagram instructions
    instead of raw content for analysis.

    Args:
        instructions: Detailed instructions describing the desired diagram
        tier: Model tier for parsing

    Returns:
        DiagramAnalysis with should_generate=True, or None on failure
    """
    try:
        result = await invoke(
            tier=tier,
            system=INSTRUCTIONS_PARSE_SYSTEM,
            user=f"""Parse these diagram instructions into structured components:

INSTRUCTIONS:
{instructions}

Extract the diagram type, title, key elements, and relationships.""",
            schema=DiagramAnalysis,
            config=InvokeConfig(max_tokens=1000),
        )
        # Custom instructions always mean we should generate
        result.should_generate = True
        logger.info(
            f"Parsed custom instructions into analysis: type={result.diagram_type}"
        )
        return result
    except Exception as e:
        logger.error(f"Failed to parse diagram instructions: {e}")
        return None


__all__ = [
    "analyze_content_for_diagram",
    "generate_svg_diagram",
    "regenerate_svg_with_feedback",
    "parse_instructions_to_analysis",
]
