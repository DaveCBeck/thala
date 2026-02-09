"""Diagram generation utilities using LLM-generated SVG.

.. deprecated::
    This pipeline (Mermaid/Graphviz/SVG → overlap check → vision select →
    refinement loop) has been replaced by direct Gemini 3 Pro image generation
    in ``workflows.shared.image_utils.generate_diagram_image()``.
    This package is retained for backward compatibility and will be removed
    in a follow-up PR after production validation.

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
from .validation import (
    extract_validation_error_type,
    sanitize_svg_text_entities,
    strip_code_fences,
    validate_and_sanitize_svg,
)
from .generation import analyze_content_for_diagram, generate_svg_diagram
from .overlap import check_bounds_violations, check_text_overlaps, check_text_shape_overlaps
from .quality_assessment import assess_diagram_quality, generate_refinement_feedback
from .refinement import refine_diagram_quality
from .graphviz_engine import generate_graphviz_diagram, generate_graphviz_with_selection
from .mermaid import generate_mermaid_diagram, generate_mermaid_with_selection
from .registry import get_available_engines, is_engine_available
from .schemas import (
    BoundsCheckResult,
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
    # Main entry points
    "generate_diagram",
    "generate_mermaid_diagram",
    "generate_mermaid_with_selection",
    "generate_graphviz_diagram",
    "generate_graphviz_with_selection",
    # Engine registry
    "get_available_engines",
    "is_engine_available",
    # Result types
    "DiagramResult",
    "DiagramCandidate",
    # Configuration
    "DiagramConfig",
    "DiagramType",
    "DiagramAnalysis",
    "OverlapCheckResult",
    "BoundsCheckResult",
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
    "check_bounds_violations",
    "check_text_shape_overlaps",
    "convert_svg_to_png",
    # Validation utilities
    "validate_and_sanitize_svg",
    "sanitize_svg_text_entities",
    "strip_code_fences",
    "extract_validation_error_type",
]
