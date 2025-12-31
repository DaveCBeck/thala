"""Synthesis subgraph for writing the final literature review.

Implements the writing pipeline:
1. Write thematic sections (one per cluster, parallel)
2. Write introduction, methodology, discussion sections
3. Integrate into coherent document
4. Process citations to Pandoc format
5. Quality verification pass

Flow:
    START -> write_intro_methodology -> write_thematic_sections
          -> write_discussion_conclusions -> integrate_sections
          -> process_citations -> verify_quality -> END
"""

from .types import QualityMetrics, SynthesisState
from .graph import synthesis_subgraph, create_synthesis_subgraph
from .api import run_synthesis

__all__ = [
    "QualityMetrics",
    "SynthesisState",
    "synthesis_subgraph",
    "create_synthesis_subgraph",
    "run_synthesis",
]
