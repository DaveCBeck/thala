"""Quality presets for synthesis workflow.

Defines quality tiers that control iteration counts, model selection,
and quality thresholds across all synthesis phases.
"""

from typing_extensions import TypedDict

from workflows.shared.quality_config import QualityTier


class SynthesisQualitySettings(TypedDict):
    """Configuration for a synthesis quality tier."""

    # Phase control
    skip_supervision: bool  # Skip supervision loop (for testing)

    # Research iterations
    web_research_runs: int  # Number of parallel web_research queries
    book_finding_runs: int  # Number of parallel book_finding themes

    # Synthesis control
    simple_synthesis: bool  # Use simplified synthesis (for testing)
    max_books_to_select: int  # Max books to select for deep integration

    # Model selection
    use_opus_for_structure: bool  # Use Opus for structure suggestion
    use_opus_for_sections: bool  # Use Opus for section writing

    # Quality thresholds
    section_quality_threshold: float  # Min quality score for sections


# Quality presets for synthesis workflow
SYNTHESIS_QUALITY_PRESETS: dict[QualityTier, SynthesisQualitySettings] = {
    "test": SynthesisQualitySettings(
        skip_supervision=True,
        web_research_runs=1,
        book_finding_runs=1,
        simple_synthesis=True,
        max_books_to_select=2,
        use_opus_for_structure=False,
        use_opus_for_sections=False,
        section_quality_threshold=0.5,
    ),
    "quick": SynthesisQualitySettings(
        skip_supervision=False,
        web_research_runs=2,
        book_finding_runs=2,
        simple_synthesis=False,
        max_books_to_select=3,
        use_opus_for_structure=True,
        use_opus_for_sections=False,
        section_quality_threshold=0.6,
    ),
    "standard": SynthesisQualitySettings(
        skip_supervision=False,
        web_research_runs=3,
        book_finding_runs=3,
        simple_synthesis=False,
        max_books_to_select=4,
        use_opus_for_structure=True,
        use_opus_for_sections=True,
        section_quality_threshold=0.7,
    ),
    "comprehensive": SynthesisQualitySettings(
        skip_supervision=False,
        web_research_runs=4,
        book_finding_runs=4,
        simple_synthesis=False,
        max_books_to_select=5,
        use_opus_for_structure=True,
        use_opus_for_sections=True,
        section_quality_threshold=0.75,
    ),
    "high_quality": SynthesisQualitySettings(
        skip_supervision=False,
        web_research_runs=5,
        book_finding_runs=5,
        simple_synthesis=False,
        max_books_to_select=6,
        use_opus_for_structure=True,
        use_opus_for_sections=True,
        section_quality_threshold=0.8,
    ),
}
