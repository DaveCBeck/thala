"""Quality tier configuration for the editing workflow."""

from typing_extensions import TypedDict


class EditingQualitySettings(TypedDict):
    """Configuration for an editing quality tier."""

    # Structure phase settings
    max_structure_iterations: int  # Max iterations for structural editing
    max_polish_edits: int  # Max polish edits to apply
    use_opus_for_analysis: bool  # Use Opus for structure analysis
    use_opus_for_generation: bool  # Use Opus for content generation
    analysis_thinking_budget: int  # Extended thinking tokens for analysis
    min_coherence_threshold: float  # Minimum coherence to pass

    # Enhancement phase settings (Phase 6)
    max_enhance_iterations: int  # Max loops within enhance phase
    enhance_word_tolerance: float  # ±tolerance for word count (e.g., 0.20 = ±20%)
    enhance_parallel_sections: int  # Max concurrent section enhancements

    # Verification phase settings (Phase 7)
    verify_use_perplexity: bool  # Enable Perplexity fact-checking
    verify_confidence_threshold: float  # Min confidence to consider resolved
    enhance_max_tool_calls: int  # Max tool calls per enhance section
    reference_check_max_tool_calls: int  # Max tool calls per reference-check (base, +5 per citation)


EDITING_QUALITY_PRESETS: dict[str, EditingQualitySettings] = {
    "test": EditingQualitySettings(
        # Structure phase
        max_structure_iterations=1,
        max_polish_edits=3,
        use_opus_for_analysis=False,  # Use Sonnet for speed
        use_opus_for_generation=False,
        analysis_thinking_budget=2000,
        min_coherence_threshold=0.6,
        # Enhancement phase
        max_enhance_iterations=1,
        enhance_word_tolerance=0.25,
        enhance_parallel_sections=3,
        # Verification phase
        verify_use_perplexity=False,  # Skip for speed in test
        verify_confidence_threshold=0.7,
        enhance_max_tool_calls=5,
        reference_check_max_tool_calls=3,
    ),
    "quick": EditingQualitySettings(
        # Structure phase
        max_structure_iterations=2,
        max_polish_edits=5,
        use_opus_for_analysis=False,
        use_opus_for_generation=False,
        analysis_thinking_budget=4000,
        min_coherence_threshold=0.7,
        # Enhancement phase
        max_enhance_iterations=2,
        enhance_word_tolerance=0.20,
        enhance_parallel_sections=5,
        # Verification phase
        verify_use_perplexity=True,
        verify_confidence_threshold=0.7,
        enhance_max_tool_calls=8,
        reference_check_max_tool_calls=5,
    ),
    "standard": EditingQualitySettings(
        # Structure phase
        max_structure_iterations=3,
        max_polish_edits=10,
        use_opus_for_analysis=True,
        use_opus_for_generation=False,
        analysis_thinking_budget=6000,
        min_coherence_threshold=0.75,
        # Enhancement phase
        max_enhance_iterations=3,
        enhance_word_tolerance=0.20,
        enhance_parallel_sections=5,
        # Verification phase
        verify_use_perplexity=True,
        verify_confidence_threshold=0.75,
        enhance_max_tool_calls=10,
        reference_check_max_tool_calls=8,
    ),
    "comprehensive": EditingQualitySettings(
        # Structure phase
        max_structure_iterations=4,
        max_polish_edits=15,
        use_opus_for_analysis=True,
        use_opus_for_generation=True,
        analysis_thinking_budget=8000,
        min_coherence_threshold=0.8,
        # Enhancement phase
        max_enhance_iterations=4,
        enhance_word_tolerance=0.15,
        enhance_parallel_sections=5,
        # Verification phase
        verify_use_perplexity=True,
        verify_confidence_threshold=0.8,
        enhance_max_tool_calls=12,
        reference_check_max_tool_calls=10,
    ),
    "high_quality": EditingQualitySettings(
        # Structure phase
        max_structure_iterations=5,
        max_polish_edits=20,
        use_opus_for_analysis=True,
        use_opus_for_generation=True,
        analysis_thinking_budget=10000,
        min_coherence_threshold=0.85,
        # Enhancement phase
        max_enhance_iterations=5,
        enhance_word_tolerance=0.10,
        enhance_parallel_sections=5,
        # Verification phase
        verify_use_perplexity=True,
        verify_confidence_threshold=0.85,
        enhance_max_tool_calls=15,
        reference_check_max_tool_calls=12,
    ),
}


def get_editing_quality_settings(quality: str) -> EditingQualitySettings:
    """Get quality settings for a tier.

    Args:
        quality: Quality tier name

    Returns:
        Quality settings dict
    """
    if quality not in EDITING_QUALITY_PRESETS:
        return EDITING_QUALITY_PRESETS["standard"]
    return EDITING_QUALITY_PRESETS[quality].copy()
