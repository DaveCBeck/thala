"""State definition and constants for diffusion engine."""

from typing import Optional
from typing_extensions import TypedDict

from workflows.academic_lit_review.state import (
    LitReviewDiffusionState,
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
    CitationEdge,
)
from workflows.academic_lit_review.citation_graph import CitationGraph
from workflows.shared.language import LanguageConfig

# Multiplier for max_papers when using non-English language verification
# We request more papers since some will be filtered by language verification
NON_ENGLISH_PAPER_OVERHEAD = 1.5

MAX_CITATIONS_PER_PAPER = 30
MAX_CONCURRENT_FETCHES = 5
COCITATION_THRESHOLD = 3


class DiffusionEngineState(TypedDict, total=False):
    """State for diffusion engine subgraph."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    discovery_seeds: list[str]
    language_config: Optional[LanguageConfig]  # For non-English paper overhead

    # Citation graph (accumulated)
    citation_graph: CitationGraph

    # Paper corpus (accumulated)
    paper_corpus: dict[str, PaperMetadata]

    # Diffusion tracking
    diffusion: LitReviewDiffusionState

    # Current stage working state
    current_stage_seeds: list[str]
    current_stage_candidates: list[PaperMetadata]
    current_stage_relevant: list[str]
    current_stage_rejected: list[str]
    new_citation_edges: list[CitationEdge]
    cocitation_included: list[str]

    # Output
    final_corpus_dois: list[str]
    saturation_reason: Optional[str]
