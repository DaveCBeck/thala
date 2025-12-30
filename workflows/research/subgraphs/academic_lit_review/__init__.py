"""Academic literature review workflow.

A multi-stage system for producing comprehensive literature reviews by:
1. Discovering seed papers via multiple strategies
2. Expanding via forward/backward citation network
3. Processing papers for structured summaries
4. Clustering into thematic groups
5. Synthesizing into a coherent review
"""

from workflows.research.subgraphs.academic_lit_review.citation_graph import (
    CitationGraph,
)
from workflows.research.subgraphs.academic_lit_review.state import (
    # Input types
    LitReviewInput,
    QualitySettings,
    QUALITY_PRESETS,
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
    # Main state
    AcademicLitReviewState,
    # Reducers
    merge_dicts,
    merge_paper_summaries,
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
    # Main state
    "AcademicLitReviewState",
    # Reducers
    "merge_dicts",
    "merge_paper_summaries",
]
