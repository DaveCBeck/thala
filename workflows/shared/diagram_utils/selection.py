"""Choose-best flow for diagram candidate selection.

Contains functions for generating multiple SVG candidates in parallel,
and using vision-based selection to choose and improve the best one.
"""

import asyncio
import base64
import logging
import re

from .conversion import convert_svg_to_png
from .generation import generate_svg_diagram
from .overlap import check_text_overlaps
from .prompts import SVG_IMPROVEMENT_SYSTEM, SVG_SELECTION_SYSTEM
from .schemas import DiagramAnalysis, DiagramCandidate, DiagramConfig

logger = logging.getLogger(__name__)


async def generate_candidates(
    analysis: DiagramAnalysis,
    config: DiagramConfig,
    num_candidates: int = 3,
) -> list[DiagramCandidate]:
    """Generate N SVG candidates in parallel, check overlaps, convert to PNG.

    Args:
        analysis: DiagramAnalysis from content analysis
        config: Diagram configuration
        num_candidates: Number of candidates to generate (default 3)

    Returns:
        List of DiagramCandidate objects with SVG, PNG, and overlap info
    """
    # Generate SVGs in parallel
    tasks = [generate_svg_diagram(analysis, config) for _ in range(num_candidates)]
    svg_results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    for i, svg_result in enumerate(svg_results):
        # Handle exceptions
        if isinstance(svg_result, Exception):
            logger.warning(f"Candidate {i + 1} generation failed: {svg_result}")
            continue

        # Handle None results
        if not svg_result:
            logger.warning(f"Candidate {i + 1} returned None")
            continue

        svg_content = svg_result

        # Check overlaps
        overlap = check_text_overlaps(svg_content)

        # Convert to PNG
        png_bytes = convert_svg_to_png(
            svg_content,
            dpi=config.dpi,
            background_color=config.background_color,
        )

        if not png_bytes:
            logger.warning(f"Candidate {i + 1} PNG conversion failed")
            continue

        candidates.append(
            DiagramCandidate(
                svg_content=svg_content,
                png_bytes=png_bytes,
                overlap_check=overlap,
                candidate_id=i + 1,
            )
        )

    logger.info(f"Generated {len(candidates)} valid candidates out of {num_candidates}")
    return candidates


def parse_selection_response(
    response: str, candidates: list[DiagramCandidate]
) -> tuple[str, int, str]:
    """Extract SVG, selected candidate ID, and rationale from response.

    Args:
        response: Full LLM response containing rationale and SVG
        candidates: List of candidates for fallback

    Returns:
        Tuple of (improved_svg, selected_candidate_id, rationale)

    Raises:
        ValueError: If no valid SVG found in response
    """
    # Find the SVG portion
    svg_start = response.find("<svg")
    svg_end = response.rfind("</svg>")

    if svg_start == -1 or svg_end < svg_start:
        raise ValueError("No valid SVG found in response")

    improved_svg = response[svg_start : svg_end + 6]
    rationale = response[:svg_start].strip()

    # Extract candidate number from rationale (e.g., "Candidate 2" or "candidate 2")
    match = re.search(r"[Cc]andidate\s*(\d)", rationale)
    selected_id = int(match.group(1)) if match else 1

    # Validate selected_id is in valid range
    valid_ids = {c.candidate_id for c in candidates}
    if selected_id not in valid_ids:
        logger.warning(f"Selected ID {selected_id} not in valid candidates, defaulting to first")
        selected_id = candidates[0].candidate_id if candidates else 1

    return improved_svg, selected_id, rationale


async def select_and_improve(
    candidates: list[DiagramCandidate],
    analysis: DiagramAnalysis,
    config: DiagramConfig,
) -> tuple[str, int, str] | None:
    """Use Sonnet with vision to select best candidate and make improvements.

    Two-phase approach:
    1. Show all candidate images + overlap analysis, ask for selection
    2. Provide selected candidate's SVG for improvement

    Args:
        candidates: List of DiagramCandidate objects
        analysis: Original diagram analysis
        config: Diagram configuration

    Returns:
        Tuple of (improved_svg, selected_id, rationale) or None on failure
    """
    from langchain_anthropic import ChatAnthropic

    if not candidates:
        logger.error("No candidates to select from")
        return None

    # If only one candidate, just return it without selection
    if len(candidates) == 1:
        logger.info("Only one candidate, skipping selection")
        return (candidates[0].svg_content, candidates[0].candidate_id, "Only one candidate available")

    try:
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8000)

        # Build candidate details for the prompt
        content_parts = []

        for candidate in candidates:
            overlap_desc = "No overlaps detected"
            if candidate.overlap_check.has_overlaps:
                pairs = candidate.overlap_check.overlap_pairs[:3]  # Limit to 3 for brevity
                overlap_desc = f"{len(candidate.overlap_check.overlap_pairs)} overlaps: " + "; ".join(
                    [f'"{t1}" / "{t2}"' for t1, t2 in pairs]
                )
                if len(candidate.overlap_check.overlap_pairs) > 3:
                    overlap_desc += f" (and {len(candidate.overlap_check.overlap_pairs) - 3} more)"

            # Add text describing this candidate
            content_parts.append({
                "type": "text",
                "text": f"**Candidate {candidate.candidate_id}**\nOverlap Analysis: {overlap_desc}",
            })

            # Add the image
            b64_png = base64.b64encode(candidate.png_bytes).decode("utf-8")
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_png,
                },
            })

        # Add final instruction
        content_parts.append({
            "type": "text",
            "text": "Which candidate is best? State your choice (1, 2, or 3) and explain briefly why.",
        })

        # Phase 1: Selection with images
        selection_response = await llm.ainvoke([
            {"role": "system", "content": SVG_SELECTION_SYSTEM},
            {"role": "user", "content": content_parts},
        ])

        selection_text = (
            selection_response.content
            if isinstance(selection_response.content, str)
            else str(selection_response.content)
        ).strip()

        # Extract selected candidate ID
        match = re.search(r"[Cc]andidate\s*(\d)", selection_text)
        selected_id = int(match.group(1)) if match else 1

        # Find the selected candidate
        selected_candidate = next(
            (c for c in candidates if c.candidate_id == selected_id),
            candidates[0],  # Fallback to first
        )

        rationale = selection_text

        # Phase 2: Improvement with SVG code
        improvement_response = await llm.ainvoke([
            {"role": "system", "content": SVG_IMPROVEMENT_SYSTEM},
            {
                "role": "user",
                "content": f"Here is the SVG for candidate {selected_id}. Make minor improvements to spacing and alignment:\n\n{selected_candidate.svg_content}",
            },
        ])

        improved_svg = (
            improvement_response.content
            if isinstance(improvement_response.content, str)
            else str(improvement_response.content)
        ).strip()

        # Clean up response - remove any markdown fences
        if improved_svg.startswith("```"):
            lines = improved_svg.split("\n")
            improved_svg = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Extract SVG if needed
        if not improved_svg.startswith("<svg"):
            svg_start = improved_svg.find("<svg")
            svg_end = improved_svg.rfind("</svg>")
            if svg_start != -1 and svg_end > svg_start:
                improved_svg = improved_svg[svg_start : svg_end + 6]
            else:
                logger.warning("Could not extract improved SVG, using original")
                improved_svg = selected_candidate.svg_content

        logger.info(f"Selected candidate {selected_id}, improved SVG ({len(improved_svg)} chars)")
        return (improved_svg, selected_id, rationale)

    except Exception as e:
        logger.error(f"Selection and improvement failed: {e}")
        # Fallback: return the candidate with fewest overlaps
        best = min(candidates, key=lambda c: len(c.overlap_check.overlap_pairs))
        return (best.svg_content, best.candidate_id, f"Fallback selection due to error: {e}")


__all__ = [
    "generate_candidates",
    "parse_selection_response",
    "select_and_improve",
]
