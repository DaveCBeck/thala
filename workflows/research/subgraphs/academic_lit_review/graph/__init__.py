"""Main graph for academic literature review workflow.

Connects all phases into a complete literature review pipeline:
1. Discovery: Find initial papers via keyword search + citation network
2. Diffusion: Expand corpus via recursive citation network exploration
3. Processing: Acquire full text, extract summaries via document_processing
4. Clustering: Dual-strategy thematic organization (BERTopic + LLM + Opus)
5. Synthesis: Write coherent literature review with proper citations

Flow:
    START -> discovery_phase -> diffusion_phase -> processing_phase
          -> clustering_phase -> synthesis_phase -> END
"""

# Configure LangSmith tracing before other imports
from core.config import configure_langsmith

configure_langsmith()

from .construction import academic_lit_review_graph, create_academic_lit_review_graph
from .api import academic_lit_review
from .phases import (
    discovery_phase_node,
    diffusion_phase_node,
    processing_phase_node,
    clustering_phase_node,
    synthesis_phase_node,
)

__all__ = [
    "academic_lit_review_graph",
    "create_academic_lit_review_graph",
    "academic_lit_review",
    "discovery_phase_node",
    "diffusion_phase_node",
    "processing_phase_node",
    "clustering_phase_node",
    "synthesis_phase_node",
]
