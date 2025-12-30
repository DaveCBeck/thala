"""
State schemas for academic literature review workflow.

Defines TypedDict states for comprehensive literature review generation through
citation network diffusion, clustering, and synthesis.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict


# =============================================================================
# Reducer Functions
# =============================================================================


def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge two dicts, with new values overwriting existing."""
    return {**existing, **new}


def merge_paper_summaries(existing: dict, new: dict) -> dict:
    """Merge paper summary dicts, preferring existing summaries if present."""
    merged = existing.copy()
    for doi, summary in new.items():
        if doi not in merged:
            merged[doi] = summary
    return merged


# =============================================================================
# Input Types
# =============================================================================


class LitReviewInput(TypedDict):
    """Input parameters for literature review workflow."""

    topic: str
    research_questions: list[str]
    quality: Literal["quick", "standard", "comprehensive", "high_quality"]
    date_range: Optional[tuple[int, int]]  # (start_year, end_year)
    include_books: bool
    focus_areas: Optional[list[str]]  # Specific sub-topics to prioritize
    exclude_terms: Optional[list[str]]  # Terms to filter out
    max_papers: Optional[int]  # Override default for quality level


class QualitySettings(TypedDict):
    """Configuration for a quality tier."""

    max_stages: int  # Maximum diffusion stages
    max_papers: int  # Maximum papers to process
    target_word_count: int  # Target length of final review
    min_citations_filter: int  # Minimum citations for discovery
    saturation_threshold: float  # Coverage delta threshold
    use_batch_api: bool  # Use Anthropic Batch API


# Quality presets mapping quality levels to settings
QUALITY_PRESETS: dict[str, QualitySettings] = {
    "quick": QualitySettings(
        max_stages=2,
        max_papers=50,
        target_word_count=3000,
        min_citations_filter=5,
        saturation_threshold=0.15,
        use_batch_api=False,
    ),
    "standard": QualitySettings(
        max_stages=3,
        max_papers=100,
        target_word_count=6000,
        min_citations_filter=10,
        saturation_threshold=0.12,
        use_batch_api=True,
    ),
    "comprehensive": QualitySettings(
        max_stages=4,
        max_papers=200,
        target_word_count=10000,
        min_citations_filter=10,
        saturation_threshold=0.10,
        use_batch_api=True,
    ),
    "high_quality": QualitySettings(
        max_stages=5,
        max_papers=300,
        target_word_count=12500,
        min_citations_filter=10,
        saturation_threshold=0.10,
        use_batch_api=True,
    ),
}


# =============================================================================
# Paper Metadata Types
# =============================================================================


class PaperAuthor(TypedDict):
    """Author metadata for a paper."""

    name: str
    author_id: Optional[str]  # OpenAlex author ID
    institution: Optional[str]
    orcid: Optional[str]


class PaperMetadata(TypedDict):
    """Core metadata for a discovered paper."""

    doi: str  # Primary identifier
    title: str
    authors: list[PaperAuthor]
    publication_date: Optional[str]
    year: int
    venue: Optional[str]  # Journal/conference name
    cited_by_count: int
    abstract: Optional[str]
    openalex_id: str
    primary_topic: Optional[str]
    is_oa: bool  # Open access
    oa_url: Optional[str]
    oa_status: Optional[str]  # "gold", "green", "hybrid", etc.
    referenced_works: list[str]  # DOIs of cited papers
    citing_works_count: int
    retrieved_at: Optional[datetime]
    discovery_stage: int  # Which diffusion stage found this
    discovery_method: str  # "keyword", "forward", "backward", "expert", "book"


class ClaimWithEvidence(TypedDict):
    """A claim extracted from a paper with evidence reference."""

    claim: str
    evidence_type: str  # "empirical", "theoretical", "methodological"
    confidence: float  # 0-1
    page_reference: Optional[str]


class PaperSummary(TypedDict):
    """Structured summary of a processed paper."""

    # Core metadata
    doi: str
    title: str
    authors: list[str]  # Author names only
    year: int
    venue: Optional[str]

    # From document_processing workflow
    short_summary: str  # 100-word summary
    es_record_id: str  # Elasticsearch document ID
    zotero_key: str  # Zotero item key

    # Extracted for clustering/synthesis
    key_findings: list[str]  # 3-5 main findings
    methodology: str  # Research method summary
    limitations: list[str]  # Stated limitations
    future_work: list[str]  # Suggested future directions
    themes: list[str]  # Topic tags
    claims: list[ClaimWithEvidence]  # Extractable claims

    # Quality metadata
    relevance_score: float  # 0-1 relevance to topic
    processing_status: str  # "success", "partial", "failed"


# =============================================================================
# Citation Network Types
# =============================================================================


class CitationEdge(TypedDict):
    """A directed citation edge in the network."""

    citing_doi: str
    cited_doi: str
    discovered_at: datetime
    edge_type: str  # "forward", "backward", "co-citation"


class PaperNode(TypedDict):
    """A node in the citation network graph."""

    doi: str
    title: str
    year: int
    cited_by_count: int
    in_degree: int  # Citations from corpus papers
    out_degree: int  # References to corpus papers
    discovery_stage: int
    cluster_id: Optional[int]


# =============================================================================
# Diffusion Algorithm Types
# =============================================================================


class DiffusionStage(TypedDict):
    """State for a single diffusion stage."""

    stage_number: int
    seed_papers: list[str]  # DOIs to expand from
    forward_papers_found: int  # Papers found via forward citations
    backward_papers_found: int  # Papers found via backward citations
    new_relevant: list[str]  # DOIs passing relevance filter
    new_rejected: list[str]  # DOIs rejected by filter
    coverage_delta: float  # Fraction of new papers that were relevant
    started_at: datetime
    completed_at: Optional[datetime]


class LitReviewDiffusionState(TypedDict):
    """State tracking for the diffusion algorithm."""

    current_stage: int
    max_stages: int
    stages: list[DiffusionStage]
    saturation_threshold: float  # Coverage delta threshold for stopping
    is_saturated: bool
    consecutive_low_coverage: int  # Stages with delta < threshold
    total_papers_discovered: int
    total_papers_relevant: int
    total_papers_rejected: int


# =============================================================================
# Clustering Types
# =============================================================================


class BERTopicCluster(TypedDict):
    """Output from BERTopic statistical clustering."""

    cluster_id: int
    topic_words: list[str]  # Representative words
    paper_dois: list[str]
    coherence_score: float


class LLMTheme(TypedDict):
    """A theme identified by LLM semantic clustering."""

    name: str
    description: str
    paper_dois: list[str]
    sub_themes: list[str]
    relationships: list[str]  # How this theme relates to others


class LLMTopicSchema(TypedDict):
    """Output from LLM semantic clustering."""

    themes: list[LLMTheme]
    reasoning: str  # LLM's clustering rationale


class ThematicCluster(TypedDict):
    """Final synthesized thematic grouping."""

    cluster_id: int
    label: str  # Final theme name
    description: str  # What this cluster covers
    paper_dois: list[str]
    key_papers: list[str]  # Most central papers
    sub_themes: list[str]  # Finer-grained topics
    conflicts: list[str]  # Contradictory findings
    gaps: list[str]  # Under-researched areas
    source: str  # "bertopic", "llm", or "merged"


# =============================================================================
# Synthesis Types
# =============================================================================


class FormattedCitation(TypedDict):
    """A formatted citation for the bibliography."""

    doi: str
    citation_text: str  # Formatted citation string
    zotero_key: str


# =============================================================================
# Main Workflow State
# =============================================================================


class AcademicLitReviewState(TypedDict):
    """Complete state for academic literature review workflow."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings

    # Discovery phase results (parallel aggregation)
    keyword_papers: Annotated[list[str], add]  # DOIs from keyword search
    citation_papers: Annotated[list[str], add]  # DOIs from citation search
    expert_papers: Annotated[list[str], add]  # DOIs from expert identification
    book_dois: Annotated[list[str], add]  # DOIs for books found

    # Diffusion tracking
    diffusion: LitReviewDiffusionState

    # Paper corpus (accumulated across diffusion stages)
    paper_corpus: Annotated[dict[str, PaperMetadata], merge_dicts]
    paper_summaries: Annotated[dict[str, PaperSummary], merge_paper_summaries]
    citation_edges: Annotated[list[CitationEdge], add]
    paper_nodes: Annotated[dict[str, PaperNode], merge_dicts]

    # Processing tracking
    papers_to_process: list[str]  # DOIs queued for processing
    papers_processed: list[str]  # DOIs successfully processed
    papers_failed: list[str]  # DOIs that failed processing

    # Clustering results
    bertopic_clusters: Optional[list[BERTopicCluster]]
    llm_topic_schema: Optional[LLMTopicSchema]
    clusters: list[ThematicCluster]

    # Synthesis outputs
    section_drafts: dict[str, str]  # Section name -> draft text
    final_review: Optional[str]  # Final integrated review
    references: list[FormattedCitation]
    prisma_documentation: Optional[str]  # Search methodology docs

    # Store integration
    elasticsearch_ids: dict[str, str]  # DOI -> ES record ID
    zotero_keys: dict[str, str]  # DOI -> Zotero key

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str  # "discovery", "diffusion", "processing", etc.
    current_status: str  # Human-readable status message
    errors: Annotated[list[dict], add]
