"""Dual-strategy clustering subgraph for thematic organization of papers.

Implements parallel clustering approaches:
1. BERTopic statistical clustering (embedding + HDBSCAN)
2. LLM semantic clustering (Sonnet 4.5 with 1M context)
3. Opus synthesis to merge both approaches into final ThematicClusters

Flow:
    START -> prepare_documents -> [parallel: bertopic_clustering, llm_clustering]
          -> synthesize_clusters -> per_cluster_analysis -> END
"""

from .analysis import ClusterAnalysis
from .api import run_clustering
from .graph import (
    ClusteringState,
    clustering_subgraph,
    create_clustering_subgraph,
    create_parallel_clustering_subgraph,
)
from .schemas import (
    ClusterAnalysisOutput,
    LLMThemeOutput,
    LLMTopicSchemaOutput,
    OpusSynthesisOutput,
    ThematicClusterOutput,
)

__all__ = [
    "LLMThemeOutput",
    "LLMTopicSchemaOutput",
    "ClusterAnalysisOutput",
    "ThematicClusterOutput",
    "OpusSynthesisOutput",
    "ClusterAnalysis",
    "ClusteringState",
    "clustering_subgraph",
    "create_clustering_subgraph",
    "create_parallel_clustering_subgraph",
    "run_clustering",
]
