"""Iterative quality refinement loop for SVG diagrams.

Repeatedly assesses diagram quality and regenerates with targeted
feedback until quality threshold is met or max iterations reached.
"""

import logging

from workflows.shared.llm_utils import ModelTier, get_llm

from .conversion import convert_svg_to_png
from .prompts import SVG_REFINEMENT_SYSTEM, SVG_REFINEMENT_USER
from .quality_assessment import assess_diagram_quality, generate_refinement_feedback
from .schemas import DiagramAnalysis, DiagramConfig, DiagramQualityAssessment

logger = logging.getLogger(__name__)


async def _regenerate_svg_with_feedback(
    svg_content: str,
    assessment: DiagramQualityAssessment,
    analysis: DiagramAnalysis,
    config: DiagramConfig,
) -> str | None:
    """Regenerate SVG based on quality feedback.

    Args:
        svg_content: Current SVG to improve
        assessment: Quality assessment with scores and issues
        analysis: Original diagram requirements
        config: Diagram configuration

    Returns:
        Improved SVG content, or None on failure
    """
    try:
        # Generate structured feedback
        priority_fixes, preserve_list = generate_refinement_feedback(assessment, analysis)

        # Build the prompt
        prompt = SVG_REFINEMENT_USER.format(
            overall_score=assessment.overall_score,
            threshold=config.quality_threshold,
            text_legibility=assessment.text_legibility,
            overlap_free=assessment.overlap_free,
            visual_hierarchy=assessment.visual_hierarchy,
            spacing_balance=assessment.spacing_balance,
            layout_logic=assessment.layout_logic,
            shape_appropriateness=assessment.shape_appropriateness,
            completeness=assessment.completeness,
            priority_fixes=priority_fixes,
            preserve_list=preserve_list,
            diagram_type=analysis.diagram_type.value,
            title=analysis.title,
            elements=", ".join(analysis.key_elements[:8]),
            relationships="; ".join(analysis.relationships[:5]),
            svg_content=svg_content,
        )

        # Use Sonnet for refinement
        llm = get_llm(tier=ModelTier.SONNET, max_tokens=8000)

        response = await llm.ainvoke([
            {"role": "system", "content": SVG_REFINEMENT_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        improved_svg = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
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
                logger.warning("Could not extract improved SVG from response")
                return None

        # Basic validation
        if not improved_svg.startswith("<svg") or not improved_svg.endswith("</svg>"):
            logger.warning("Invalid SVG format in response")
            return None

        logger.info(f"Generated improved SVG ({len(improved_svg)} chars)")
        return improved_svg

    except Exception as e:
        logger.error(f"SVG regeneration failed: {e}")
        return None


async def refine_diagram_quality(
    svg_content: str,
    analysis: DiagramAnalysis,
    config: DiagramConfig,
) -> tuple[str, DiagramQualityAssessment | None, list[float]]:
    """Iteratively refine diagram until quality threshold met.

    Loop logic:
    1. Convert SVG to PNG
    2. Assess quality with vision model
    3. If meets threshold or max iterations reached, exit
    4. If no improvement for 2 consecutive rounds, exit
    5. Generate feedback and regenerate SVG
    6. Loop back to step 1

    Args:
        svg_content: Initial SVG to refine
        analysis: Original diagram analysis
        config: Configuration with threshold and max iterations

    Returns:
        Tuple of (final_svg, final_assessment, quality_history)
        - final_svg: Best SVG achieved
        - final_assessment: Assessment of final SVG (or None if all failed)
        - quality_history: List of scores at each iteration
    """
    max_iterations = config.max_refinement_iterations
    threshold = config.quality_threshold

    current_svg = svg_content
    quality_history: list[float] = []
    best_svg = svg_content
    best_score = 0.0
    best_assessment: DiagramQualityAssessment | None = None
    consecutive_no_improvement = 0

    logger.info(
        f"Starting quality refinement loop (max={max_iterations}, threshold={threshold})"
    )

    for iteration in range(max_iterations):
        logger.info(f"Refinement iteration {iteration + 1}/{max_iterations}")

        # Convert SVG to PNG for assessment
        png_bytes = convert_svg_to_png(
            current_svg,
            dpi=config.dpi,
            background_color=config.background_color,
        )

        if not png_bytes:
            logger.warning(f"PNG conversion failed at iteration {iteration + 1}")
            break

        # Assess quality
        assessment = await assess_diagram_quality(
            svg_content=current_svg,
            png_bytes=png_bytes,
            analysis=analysis,
            config=config,
        )

        if not assessment:
            logger.warning(f"Assessment failed at iteration {iteration + 1}")
            break

        current_score = assessment.overall_score
        quality_history.append(current_score)

        logger.info(
            f"Iteration {iteration + 1}: score={current_score:.2f} "
            f"(best={best_score:.2f}, threshold={threshold})"
        )

        # Track best result
        if current_score > best_score:
            best_svg = current_svg
            best_score = current_score
            best_assessment = assessment
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        # Exit if threshold met
        if assessment.meets_threshold:
            logger.info(f"Quality threshold met at iteration {iteration + 1}")
            return current_svg, assessment, quality_history

        # Exit if no improvement for 2 consecutive rounds
        if consecutive_no_improvement >= 2:
            logger.info(
                f"No improvement for 2 rounds, stopping at iteration {iteration + 1}"
            )
            break

        # Don't regenerate on last iteration
        if iteration >= max_iterations - 1:
            break

        # Regenerate with feedback
        improved_svg = await _regenerate_svg_with_feedback(
            svg_content=current_svg,
            assessment=assessment,
            analysis=analysis,
            config=config,
        )

        if not improved_svg:
            logger.warning(f"Regeneration failed at iteration {iteration + 1}")
            break

        current_svg = improved_svg

    # Return best result achieved
    final_assessment = best_assessment
    if quality_history:
        logger.info(
            f"Refinement complete: {len(quality_history)} iterations, "
            f"best score={best_score:.2f}"
        )
    else:
        logger.warning("Refinement produced no quality history")

    return best_svg, final_assessment, quality_history


__all__ = [
    "refine_diagram_quality",
]
