"""Academic literature review workflow.

A multi-stage system for producing comprehensive literature reviews by:
1. Discovering seed papers via multiple strategies
2. Expanding via forward/backward citation network
3. Processing papers for structured summaries
4. Clustering into thematic groups
5. Synthesizing into a coherent review
"""

from workflows.research.academic_lit_review.citation_graph import (
    CitationGraph,
)
from workflows.research.academic_lit_review.state import (
    # Input types
    LitReviewInput,
    # Paper types
    PaperAuthor,
    PaperMetadata,
    PaperSummary,
    ClaimWithEvidence,
    # Citation network types
    CitationEdge,
    PaperNode,
    # Diffusion types
    DiffusionStage,
    LitReviewDiffusionState,
    # Clustering types
    BERTopicCluster,
    LLMTheme,
    LLMTopicSchema,
    ThematicCluster,
    # Synthesis types
    FormattedCitation,
    # Supervision types
    SupervisionState,
    SupervisionExpansion,
    # Main state
    AcademicLitReviewState,
)
from workflows.research.academic_lit_review.quality_presets import (
    QualitySettings,
    QUALITY_PRESETS,
)
from workflows.research.academic_lit_review.reducers import (
    merge_dicts,
    merge_paper_summaries,
)
from workflows.research.academic_lit_review.utils import (
    convert_to_paper_metadata,
    deduplicate_papers,
    score_paper_relevance,
    batch_score_relevance,
    generate_search_queries,
)
from workflows.research.academic_lit_review.keyword_search import (
    KeywordSearchState,
    keyword_search_subgraph,
    run_keyword_search,
)
from workflows.research.academic_lit_review.citation_network import (
    CitationNetworkState,
    citation_network_subgraph,
    run_citation_expansion,
)
from workflows.research.academic_lit_review.diffusion_engine import (
    DiffusionEngineState,
    diffusion_engine_subgraph,
    run_diffusion,
)
from workflows.research.academic_lit_review.paper_processor import (
    PaperProcessingState,
    paper_processing_subgraph,
    run_paper_processing,
)
from workflows.research.academic_lit_review.clustering import (
    ClusteringState,
    ClusterAnalysis,
    clustering_subgraph,
    run_clustering,
)
from workflows.research.academic_lit_review.synthesis import (
    SynthesisState,
    QualityMetrics,
    synthesis_subgraph,
    run_synthesis,
)
from workflows.research.academic_lit_review.graph import (
    academic_lit_review_graph,
    academic_lit_review,
)

__all__ = [
    # Citation graph
    "CitationGraph",
    # Input types
    "LitReviewInput",
    "QualitySettings",
    "QUALITY_PRESETS",
    # Paper types
    "PaperAuthor",
    "PaperMetadata",
    "PaperSummary",
    "ClaimWithEvidence",
    # Citation network types
    "CitationEdge",
    "PaperNode",
    # Diffusion types
    "DiffusionStage",
    "LitReviewDiffusionState",
    # Clustering types
    "BERTopicCluster",
    "LLMTheme",
    "LLMTopicSchema",
    "ThematicCluster",
    # Synthesis types
    "FormattedCitation",
    # Supervision types
    "SupervisionState",
    "SupervisionExpansion",
    # Main state
    "AcademicLitReviewState",
    # Reducers
    "merge_dicts",
    "merge_paper_summaries",
    # Utils
    "convert_to_paper_metadata",
    "deduplicate_papers",
    "score_paper_relevance",
    "batch_score_relevance",
    "generate_search_queries",
    # Keyword search subgraph
    "KeywordSearchState",
    "keyword_search_subgraph",
    "run_keyword_search",
    # Citation network subgraph
    "CitationNetworkState",
    "citation_network_subgraph",
    "run_citation_expansion",
    # Diffusion engine subgraph
    "DiffusionEngineState",
    "diffusion_engine_subgraph",
    "run_diffusion",
    # Paper processor subgraph
    "PaperProcessingState",
    "paper_processing_subgraph",
    "run_paper_processing",
    # Clustering subgraph
    "ClusteringState",
    "ClusterAnalysis",
    "clustering_subgraph",
    "run_clustering",
    # Synthesis subgraph
    "SynthesisState",
    "QualityMetrics",
    "synthesis_subgraph",
    "run_synthesis",
    # Main workflow
    "academic_lit_review_graph",
    "academic_lit_review",
]
