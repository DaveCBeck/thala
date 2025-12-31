"""Type definitions for synthesis subgraph."""

from typing import Optional

from typing_extensions import TypedDict

from workflows.research.subgraphs.academic_lit_review.state import (
    FormattedCitation,
    LitReviewInput,
    PaperSummary,
    QualitySettings,
    ThematicCluster,
)
from workflows.research.subgraphs.academic_lit_review.clustering import ClusterAnalysis

MAX_CONCURRENT_SECTIONS = 5
TARGET_SECTION_WORDS = 1500
MIN_CITATIONS_PER_SECTION = 5


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
