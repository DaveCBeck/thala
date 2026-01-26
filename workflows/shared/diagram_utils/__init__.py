"""Diagram generation utilities using LLM-generated SVG.

Generates relevant diagrams from content using a multi-stage pipeline:
1. Analyze content to determine if/how to create a diagram
2. Generate multiple SVG candidates in parallel
3. Check text overlaps and convert to PNG for each candidate
4. Use vision model to select best candidate and improve it
5. (Optional) Iterative quality refinement loop
6. Convert final improved SVG to PNG

Example:
    result = await generate_diagram(
        title="Paper Processing Pipeline",
        content="The pipeline has three stages: extraction, validation...",
    )
    if result.success:
        with open("diagram.png", "wb") as f:
            f.write(result.png_bytes)

Example with quality refinement:
    result = await generate_diagram(
        title="Process Flow",
        content="...",
        config=DiagramConfig(
            enable_refinement_loop=True,
            quality_threshold=3.5,
            max_refinement_iterations=3,
        ),
    )
    if result.refinement_iterations:
        print(f"Refined in {result.refinement_iterations} iterations")
        print(f"Final quality: {result.final_quality_score}")
"""

# Re-export all public symbols for backward compatibility
from .conversion import convert_svg_to_png
from .core import generate_diagram
from .generation import analyze_content_for_diagram, generate_svg_diagram
from .overlap import check_text_overlaps
from .quality_assessment import assess_diagram_quality, generate_refinement_feedback
from .refinement import refine_diagram_quality
from .schemas import (
    DiagramAnalysis,
    DiagramCandidate,
    DiagramConfig,
    DiagramQualityAssessment,
    DiagramResult,
    DiagramType,
    OverlapCheckResult,
    QualityIssue,
)

__all__ = [
    # Main entry point
    "generate_diagram",
    # Result types
    "DiagramResult",
    "DiagramCandidate",
    # Configuration
    "DiagramConfig",
    "DiagramType",
    "DiagramAnalysis",
    "OverlapCheckResult",
    # Quality assessment
    "DiagramQualityAssessment",
    "QualityIssue",
    "assess_diagram_quality",
    "generate_refinement_feedback",
    "refine_diagram_quality",
    # Individual pipeline functions
    "analyze_content_for_diagram",
    "generate_svg_diagram",
    "check_text_overlaps",
    "convert_svg_to_png",
]
