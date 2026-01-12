"""Types and constants for citation network discovery."""

from typing import Optional
from typing_extensions import TypedDict

from workflows.academic_lit_review.state import (
    CitationEdge,
    LitReviewInput,
    PaperMetadata,
)
from workflows.academic_lit_review.quality_presets import QualitySettings

# Constants
MAX_CITATIONS_PER_PAPER = 30
MAX_CONCURRENT_FETCHES = 5


class CitationNetworkState(TypedDict):
    """State for citation-based discovery subgraph."""

    # Input (from parent)
    input: LitReviewInput
    quality_settings: QualitySettings
    seed_dois: list[str]
    existing_dois: set[str]
    language_config: Optional[dict]

    # Internal state
    forward_results: list[dict]
    backward_results: list[dict]
    citation_edges: list[CitationEdge]

    # Output
    discovered_papers: list[PaperMetadata]
    rejected_papers: list[PaperMetadata]
    discovered_dois: list[str]
    new_edges: list[CitationEdge]
