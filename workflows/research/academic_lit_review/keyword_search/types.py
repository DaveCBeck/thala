"""Types and state for keyword search subgraph."""

from typing import Optional
from typing_extensions import TypedDict

from langchain_tools.openalex import OpenAlexWork
from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    PaperMetadata,
    QualitySettings,
)
from workflows.shared.language import LanguageConfig

MAX_QUERIES = 5
MAX_RESULTS_PER_QUERY = 25


class KeywordSearchState(TypedDict):
    """State for keyword-based academic search subgraph."""

    # Input (from parent)
    input: LitReviewInput
    quality_settings: QualitySettings
    language_config: Optional[LanguageConfig]

    # Internal state
    search_queries: list[str]
    raw_results: list[OpenAlexWork]

    # Output
    discovered_papers: list[PaperMetadata]
    rejected_papers: list[PaperMetadata]
    keyword_dois: list[str]
