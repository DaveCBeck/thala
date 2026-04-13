"""Quality tier configuration for academic literature review workflow."""

from typing_extensions import TypedDict


class QualitySettings(TypedDict):
    """Configuration for a quality tier."""

    max_stages: int  # Maximum diffusion stages
    max_papers: int  # Maximum papers to process
    target_word_count: int  # Target length of final review
    min_citations_filter: int  # Minimum citations for discovery
    saturation_threshold: float  # Coverage delta threshold
    supervision_loops: str  # Which supervision loops to run: "none", "one", "two", "three", "four", "all"
    recency_years: int  # Primary recent window in years (default: 1 → 2025+)
    recency_years_fallback: int  # Fallback if primary yields too few (default: 2 → 2024+)
    recency_quota: float  # Target fraction of recent papers (default: 0.35)
    use_opus_for_writing: bool  # Use Opus for drafting, revision, and integration


QUALITY_PRESETS: dict[str, QualitySettings] = {
    "test": QualitySettings(
        max_stages=1,
        max_papers=5,
        target_word_count=2000,
        min_citations_filter=0,
        saturation_threshold=0.25,
        supervision_loops="none",
        recency_years=1,
        recency_years_fallback=2,
        recency_quota=0.0,  # Skip recency quota for test tier
        use_opus_for_writing=False,
    ),
    "quick": QualitySettings(
        max_stages=2,
        max_papers=30,
        target_word_count=9000,
        min_citations_filter=5,
        saturation_threshold=0.15,
        supervision_loops="all",
        recency_years=1,
        recency_years_fallback=2,
        recency_quota=0.35,
        use_opus_for_writing=False,
    ),
    "standard": QualitySettings(
        max_stages=3,
        max_papers=50,
        target_word_count=12000,
        min_citations_filter=10,
        saturation_threshold=0.12,
        supervision_loops="all",
        recency_years=1,
        recency_years_fallback=2,
        recency_quota=0.35,
        use_opus_for_writing=True,
    ),
    "comprehensive": QualitySettings(
        max_stages=4,
        max_papers=100,
        target_word_count=17500,
        min_citations_filter=10,
        saturation_threshold=0.10,
        supervision_loops="all",
        recency_years=1,
        recency_years_fallback=2,
        recency_quota=0.35,
        use_opus_for_writing=True,
    ),
    "high_quality": QualitySettings(
        max_stages=5,
        max_papers=250,
        target_word_count=22000,
        min_citations_filter=10,
        saturation_threshold=0.10,
        supervision_loops="all",
        recency_years=1,
        recency_years_fallback=2,
        recency_quota=0.35,
        use_opus_for_writing=True,
    ),
}
