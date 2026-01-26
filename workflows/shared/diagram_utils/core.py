"""Core diagram generation pipeline.

Contains the main entry point for generating diagrams from content.
"""

import logging

from .conversion import convert_svg_to_png
from .generation import analyze_content_for_diagram, parse_instructions_to_analysis
from .overlap import check_text_overlaps
from .schemas import DiagramConfig, DiagramResult, DiagramType
from .selection import generate_candidates, select_and_improve

logger = logging.getLogger(__name__)


async def generate_diagram(
    title: str,
    content: str,
    config: DiagramConfig | None = None,
    force_type: DiagramType | None = None,
    custom_instructions: str | None = None,
) -> DiagramResult:
    """Generate a diagram from content (main entry point).

    Full pipeline:
    1. Analyze content to determine diagram type and elements
    2. Generate multiple SVG candidates in parallel
    3. Check text overlaps and convert to PNG for each candidate
    4. Use vision model to select best candidate and improve it
    5. Convert final improved SVG to PNG

    Args:
        title: Content title
        content: Full content text
        config: Optional configuration (dimensions, colors, etc.)
        force_type: Force a specific diagram type (skip analysis decision)
        custom_instructions: If provided, parse these instructions into a
            DiagramAnalysis instead of analyzing the content. Use this when
            you have detailed diagram specifications from an external source.

    Returns:
        DiagramResult with SVG bytes, PNG bytes, and metadata

    Example:
        result = await generate_diagram(
            title="Paper Processing Pipeline",
            content="The pipeline has three stages: extraction, validation...",
        )
        if result.success:
            with open("diagram.png", "wb") as f:
                f.write(result.png_bytes)

        # With custom instructions (skips content analysis):
        result = await generate_diagram(
            title="",
            content="",
            custom_instructions="Create a flowchart showing: Input -> Processing -> Output",
        )
    """
    config = config or DiagramConfig()

    # Stage 1: Parse custom instructions or analyze content
    if custom_instructions:
        logger.info("Using custom instructions instead of content analysis")
        analysis = await parse_instructions_to_analysis(custom_instructions)
    else:
        analysis = await analyze_content_for_diagram(title, content)

    if not analysis:
        return DiagramResult(
            svg_bytes=None,
            png_bytes=None,
            analysis=None,
            overlap_check=None,
            generation_attempts=0,
            success=False,
            error="Content analysis failed",
        )

    if not analysis.should_generate and not force_type:
        return DiagramResult(
            svg_bytes=None,
            png_bytes=None,
            analysis=analysis,
            overlap_check=None,
            generation_attempts=0,
            success=False,
            error=f"Analysis determined diagram not needed: {analysis.rationale}",
        )

    # Override type if forced
    if force_type:
        analysis.diagram_type = force_type

    # Stage 2: Generate multiple candidates in parallel
    candidates = await generate_candidates(
        analysis=analysis,
        config=config,
        num_candidates=config.num_candidates,
    )

    if not candidates:
        return DiagramResult(
            svg_bytes=None,
            png_bytes=None,
            analysis=analysis,
            overlap_check=None,
            generation_attempts=config.num_candidates,
            success=False,
            error="All SVG generation attempts failed",
        )

    # Stage 3: Select best candidate and improve using vision
    selection_result = await select_and_improve(
        candidates=candidates,
        analysis=analysis,
        config=config,
    )

    if not selection_result:
        # Fallback: use the candidate with fewest overlaps
        best_candidate = min(candidates, key=lambda c: len(c.overlap_check.overlap_pairs))
        svg_content = best_candidate.svg_content
        selected_id = best_candidate.candidate_id
        rationale = "Fallback: selected candidate with fewest overlaps"
    else:
        svg_content, selected_id, rationale = selection_result

    # Stage 4: Iterative quality refinement (if enabled)
    refinement_iterations = None
    final_quality_score = None
    quality_history = None

    if config.enable_refinement_loop:
        from .refinement import refine_diagram_quality

        logger.info("Starting iterative quality refinement")
        svg_content, final_assessment, quality_history = await refine_diagram_quality(
            svg_content=svg_content,
            analysis=analysis,
            config=config,
        )
        refinement_iterations = len(quality_history) if quality_history else 0
        if final_assessment:
            final_quality_score = final_assessment.overall_score
            logger.info(
                f"Refinement complete: {refinement_iterations} iterations, "
                f"final score={final_quality_score:.2f}"
            )

    # Stage 5: Final overlap check and PNG conversion
    final_overlap_check = check_text_overlaps(svg_content)
    svg_bytes = svg_content.encode("utf-8")
    png_bytes = convert_svg_to_png(
        svg_content,
        dpi=config.dpi,
        background_color=config.background_color,
    )

    # Determine improvements made by comparing to original candidate
    improvements_made = []
    original_candidate = next(
        (c for c in candidates if c.candidate_id == selected_id),
        None,
    )
    if original_candidate and svg_content != original_candidate.svg_content:
        improvements_made.append("SVG was modified during improvement phase")
        if original_candidate.overlap_check.has_overlaps and not final_overlap_check.has_overlaps:
            improvements_made.append("Overlaps were fixed")
        elif (
            original_candidate.overlap_check.has_overlaps
            and final_overlap_check.has_overlaps
            and len(final_overlap_check.overlap_pairs) < len(original_candidate.overlap_check.overlap_pairs)
        ):
            improvements_made.append(
                f"Overlaps reduced from {len(original_candidate.overlap_check.overlap_pairs)} to {len(final_overlap_check.overlap_pairs)}"
            )

    has_unresolved_overlaps = final_overlap_check and final_overlap_check.has_overlaps
    error_msg = None
    if has_unresolved_overlaps:
        error_msg = f"Generated with {len(final_overlap_check.overlap_pairs)} unresolved text overlaps"

    return DiagramResult(
        svg_bytes=svg_bytes,
        png_bytes=png_bytes,
        analysis=analysis,
        overlap_check=final_overlap_check,
        generation_attempts=len(candidates),
        selected_candidate=selected_id,
        selection_rationale=rationale,
        improvements_made=improvements_made if improvements_made else None,
        success=True,
        error=error_msg,
        refinement_iterations=refinement_iterations,
        final_quality_score=final_quality_score,
        quality_history=quality_history,
    )


__all__ = [
    "generate_diagram",
]
