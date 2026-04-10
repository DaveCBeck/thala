"""Type definitions for synthesis subgraph."""

from typing import Optional

from typing_extensions import TypedDict

from workflows.research.academic_lit_review.state import (
    FormattedCitation,
    LitReviewInput,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.research.academic_lit_review.clustering import ClusterAnalysis

MAX_CONCURRENT_SECTIONS = 5
TARGET_SECTION_WORDS = 1500
MIN_CITATIONS_PER_SECTION = 5


class DiffusionStageReport(TypedDict):
    """Per-stage diffusion metrics for transparency reporting."""

    stage_number: int
    forward_found: int
    backward_found: int
    new_relevant: int
    new_rejected: int
    coverage_delta: float


class TransparencyReport(TypedDict, total=False):
    """Aggregated transparency data for honest methodology writing.

    All fields are optional (total=False) to handle older workflow states
    that may be loaded from checkpoints before these fields existed.
    """

    # Discovery
    search_queries: list[str]
    keyword_paper_count: int
    citation_paper_count: int
    expert_paper_count: int
    raw_results_count: int

    # Diffusion
    diffusion_stages: list[DiffusionStageReport]
    total_discovered: int
    total_rejected: int
    saturation_reason: str

    # Quality filters applied
    min_citations_filter: int
    recency_years: int
    recency_quota: float
    # TODO: extract from QualitySettings when relevance_threshold is added there
    relevance_threshold: float

    # Processing
    papers_processed_count: int
    papers_failed_count: int
    metadata_only_count: int
    fallback_substitutions_count: int
    fallback_substitutions_summary: str
    fallback_exhausted_count: int

    # Clustering
    clustering_method: str
    clustering_rationale: str
    cluster_count: int

    # Corpus composition
    date_range: str
    total_corpus_size: int


class QualityMetrics(TypedDict):
    """Quality metrics for the literature review."""

    total_words: int
    citation_count: int
    unique_papers_cited: int
    corpus_coverage: float
    uncited_papers: list[str]
    sections_count: int
    avg_section_length: int
    issues: list[str]


class SynthesisState(TypedDict):
    """State for synthesis/writing subgraph."""

    input: LitReviewInput
    quality_settings: QualitySettings
    paper_summaries: dict[str, PaperSummary]
    clusters: list[ThematicCluster]
    cluster_analyses: list[ClusterAnalysis]
    zotero_keys: dict[str, str]
    transparency_report: Optional[TransparencyReport]
    editorial_stance: Optional[str]
    introduction_draft: str
    methodology_draft: str
    thematic_section_drafts: dict[str, str]
    discussion_draft: str
    conclusions_draft: str
    integrated_review: str
    final_review: str
    references: list[FormattedCitation]
    citation_keys: list[str]
    quality_metrics: Optional[QualityMetrics]
    quality_passed: bool
    prisma_documentation: str
