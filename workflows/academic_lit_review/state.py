"""
State schemas for academic literature review workflow.

Defines TypedDict states for comprehensive literature review generation through
citation network diffusion, clustering, and synthesis.
"""

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from workflows.shared.language import LanguageConfig
from workflows.academic_lit_review.reducers import merge_dicts, merge_paper_summaries
from workflows.academic_lit_review.quality_presets import QualitySettings


# =============================================================================
# Input Types
# =============================================================================


class LitReviewInput(TypedDict):
    """Input parameters for literature review workflow."""

    topic: str
    research_questions: list[str]
    quality: Literal["test", "quick", "standard", "comprehensive", "high_quality"]
    date_range: Optional[tuple[int, int]]  # (start_year, end_year)
    include_books: bool
    focus_areas: Optional[list[str]]  # Specific sub-topics to prioritize
    exclude_terms: Optional[list[str]]  # Terms to filter out
    max_papers: Optional[int]  # Override default for quality level
    language_code: Optional[str]  # ISO 639-1 code, default "en"


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
    relevance_score: Optional[float]  # LLM-assigned relevance to research topic (0-1)


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
# Supervision Types
# =============================================================================


class SupervisionExpansion(TypedDict):
    """Record of a single supervision expansion iteration."""

    iteration: int
    topic: str
    issue_type: str
    research_query: str
    papers_added: list[str]  # DOIs of papers added to corpus
    integration_summary: str  # What was integrated and how


class RevisionRecord(TypedDict):
    """Record of what each loop changed."""

    loop_number: int
    iteration: int
    summary: str  # Haiku-generated summary
    changes_made: list[str]
    reasoning: str


class MultiLoopProgress(TypedDict):
    """Progress tracking across all supervision loops."""

    current_loop: int  # 1-5
    loop_iterations: dict[str, int]  # "loop_1" -> iterations used (string keys for TypedDict)
    max_iterations_per_loop: int  # Budget per loop (not shared across loops)
    revision_history: list[RevisionRecord]
    loop3_repeat_count: int  # For Loop 4.5 -> Loop 3 return (max 1)


class SupervisionState(TypedDict):
    """State tracking for supervision loop execution."""

    iteration: int
    max_iterations: int
    supervision_depth: int  # Prevents recursive supervision (max: 2)
    current_review: str     # Evolving review text
    issues_explored: list[str]  # Topics already explored (prevent re-exploration)
    is_complete: bool       # True when pass_through or max iterations reached
    loop_progress: MultiLoopProgress
    human_review_items: list[str]


# =============================================================================
# Main Workflow State
# =============================================================================


class AcademicLitReviewState(TypedDict):
    """Complete state for academic literature review workflow."""

    # Input
    input: LitReviewInput
    quality_settings: QualitySettings
    language_config: Optional[LanguageConfig]

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
    final_report: Optional[str]  # Alias for final_review (standardized field name)
    references: list[FormattedCitation]
    prisma_documentation: Optional[str]  # Search methodology docs

    # Supervision phase outputs
    supervision: Optional[SupervisionState]
    final_review_v2: Optional[str]  # Review after supervision iterations
    supervision_expansions: Annotated[list[SupervisionExpansion], add]

    # Store integration
    elasticsearch_ids: dict[str, str]  # DOI -> ES record ID
    zotero_keys: dict[str, str]  # DOI -> Zotero key

    # Workflow metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    current_phase: str  # "discovery", "diffusion", "processing", etc.
    current_status: str  # Human-readable status message
    status: Optional[str]  # Standardized: "success", "partial", "failed"
    langsmith_run_id: Optional[str]  # LangSmith run ID for tracing
    errors: Annotated[list[dict], add]
