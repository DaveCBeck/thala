"""Quality assessment for SVG diagrams using vision model.

Evaluates diagram visual quality on 7 criteria and provides
structured feedback for iterative refinement.
"""

import base64
import logging

from workflows.shared.llm_utils import ModelTier, get_structured_output

from .overlap import check_bounds_violations, check_text_overlaps, check_text_shape_overlaps
from .prompts import DIAGRAM_QUALITY_SYSTEM, DIAGRAM_QUALITY_USER
from .schemas import DiagramAnalysis, DiagramConfig, DiagramQualityAssessment

logger = logging.getLogger(__name__)


async def assess_diagram_quality(
    svg_content: str,
    png_bytes: bytes,
    analysis: DiagramAnalysis,
    config: DiagramConfig,
) -> DiagramQualityAssessment | None:
    """Use vision model to evaluate diagram quality.

    Runs programmatic overlap check and sends diagram image to vision model
    for quality assessment on 7 criteria.

    Args:
        svg_content: Current SVG content (for overlap check)
        png_bytes: PNG rendering of the diagram
        analysis: Original diagram analysis (type, elements, relationships)
        config: Configuration with quality threshold

    Returns:
        DiagramQualityAssessment with scores and issues, or None on failure
    """
    try:
        # Run programmatic checks to inform the LLM
        overlap_result = check_text_overlaps(svg_content)
        bounds_result = check_bounds_violations(svg_content)
        text_shape_overlaps = check_text_shape_overlaps(svg_content)

        # Build comprehensive report for the LLM
        report_sections = []

        # Text-text overlaps
        if overlap_result.has_overlaps:
            section = f"⚠️ TEXT OVERLAPS ({len(overlap_result.overlap_pairs)} found):\n"
            for t1, t2 in overlap_result.overlap_pairs[:5]:
                section += f'  - "{t1}" overlaps with "{t2}"\n'
            if len(overlap_result.overlap_pairs) > 5:
                section += f"  - ... and {len(overlap_result.overlap_pairs) - 5} more"
            report_sections.append(section)
        else:
            report_sections.append("✓ No text-text overlaps detected.")

        # Text-shape overlaps (circles/dots over text)
        if text_shape_overlaps:
            section = f"⚠️ TEXT OBSCURED BY SHAPES ({len(text_shape_overlaps)} found):\n"
            for desc in text_shape_overlaps[:5]:
                section += f"  - {desc}\n"
            if len(text_shape_overlaps) > 5:
                section += f"  - ... and {len(text_shape_overlaps) - 5} more"
            report_sections.append(section)
        else:
            report_sections.append("✓ No text obscured by shapes.")

        # Bounds violations (text cut off at edges)
        if bounds_result.has_violations:
            section = f"⚠️ ELEMENTS EXCEED BOUNDS ({len(bounds_result.violations)} found):\n"
            for desc in bounds_result.violations[:5]:
                section += f"  - {desc}\n"
            if len(bounds_result.violations) > 5:
                section += f"  - ... and {len(bounds_result.violations) - 5} more"
            report_sections.append(section)
        else:
            report_sections.append("✓ All elements within SVG bounds.")

        overlap_report = "\n".join(report_sections)

        # Encode PNG for vision
        b64_image = base64.b64encode(png_bytes).decode("utf-8")

        # Build multi-part content for vision
        content = [
            {
                "type": "text",
                "text": DIAGRAM_QUALITY_USER.format(
                    diagram_type=analysis.diagram_type.value,
                    title=analysis.title,
                    elements=", ".join(analysis.key_elements[:8]),
                    relationships="; ".join(analysis.relationships[:5]),
                    overlap_report=overlap_report,
                    threshold=config.quality_threshold,
                ),
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_image,
                },
            },
        ]

        # Get structured quality assessment
        assessment = await get_structured_output(
            output_schema=DiagramQualityAssessment,
            user_prompt=content,
            system_prompt=DIAGRAM_QUALITY_SYSTEM,
            tier=ModelTier.SONNET,
            max_tokens=2000,
        )

        # Calculate overall score if not provided correctly
        scores = [
            assessment.text_legibility,
            assessment.overlap_free,
            assessment.visual_hierarchy,
            assessment.spacing_balance,
            assessment.layout_logic,
            assessment.shape_appropriateness,
            assessment.completeness,
        ]
        calculated_score = sum(scores) / len(scores)

        # Use calculated score if model's score seems off
        if abs(assessment.overall_score - calculated_score) > 0.5:
            logger.debug(
                f"Correcting overall_score from {assessment.overall_score} to {calculated_score}"
            )
            assessment.overall_score = round(calculated_score, 2)

        # Ensure meets_threshold is correct
        assessment.meets_threshold = assessment.overall_score >= config.quality_threshold

        logger.info(
            f"Quality assessment: {assessment.overall_score:.2f}/5.0 "
            f"(threshold={config.quality_threshold}, meets={assessment.meets_threshold})"
        )

        return assessment

    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return None


def generate_refinement_feedback(
    assessment: DiagramQualityAssessment,
    analysis: DiagramAnalysis,
) -> tuple[str, str]:
    """Convert quality assessment into actionable feedback for refinement.

    Args:
        assessment: Quality assessment with scores and issues
        analysis: Original diagram analysis

    Returns:
        Tuple of (priority_fixes, preserve_list) as formatted strings
    """
    # Identify what's working well (scores >= 4)
    preserve = []
    if assessment.text_legibility >= 4:
        preserve.append("Text legibility is good")
    if assessment.overlap_free >= 4:
        preserve.append("Element spacing is clean")
    if assessment.visual_hierarchy >= 4:
        preserve.append("Visual hierarchy is clear")
    if assessment.spacing_balance >= 4:
        preserve.append("Whitespace balance is good")
    if assessment.layout_logic >= 4:
        preserve.append("Layout logic is sound")
    if assessment.shape_appropriateness >= 4:
        preserve.append("Shape choices are appropriate")
    if assessment.completeness >= 4:
        preserve.append("All key elements are present")

    if not preserve:
        preserve.append("General diagram structure")

    # Sort issues by severity and convert to priority fixes
    severity_order = {"severe": 0, "moderate": 1, "minor": 2}
    sorted_issues = sorted(
        assessment.issues,
        key=lambda x: severity_order.get(x.severity, 2),
    )

    priority_fixes = []
    for i, issue in enumerate(sorted_issues[:5], 1):
        fix_text = f"{i}. [{issue.severity.upper()}] {issue.category}: {issue.description}"
        if issue.suggested_fix:
            fix_text += f"\n   Fix: {issue.suggested_fix}"
        priority_fixes.append(fix_text)

    if not priority_fixes:
        # Generate fixes from low scores
        score_fixes = []
        if assessment.text_legibility < 3:
            score_fixes.append("1. Improve text legibility - increase font sizes, improve contrast")
        if assessment.overlap_free < 3:
            score_fixes.append("2. Fix overlapping elements - increase spacing between text and shapes")
        if assessment.visual_hierarchy < 3:
            score_fixes.append("3. Improve visual hierarchy - make title larger, primary elements more prominent")
        if assessment.spacing_balance < 3:
            score_fixes.append("4. Balance spacing - distribute whitespace more evenly")
        if assessment.layout_logic < 3:
            score_fixes.append("5. Improve layout logic - arrange elements to follow natural reading flow")
        priority_fixes = score_fixes[:3] if score_fixes else ["No specific fixes identified"]

    return "\n".join(priority_fixes), "\n".join(f"- {p}" for p in preserve)


__all__ = [
    "assess_diagram_quality",
    "generate_refinement_feedback",
]
