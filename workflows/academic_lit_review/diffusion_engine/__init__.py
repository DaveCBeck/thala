"""Diffusion engine subgraph for iterative corpus expansion through citation network.

Implements multi-stage citation diffusion with two-stage relevance filtering:
1. Co-citation analysis (automatic inclusion for highly co-cited papers)
2. LLM-based relevance scoring for remaining candidates

Flow:
    START -> initialize -> select_seeds -> citation_expansion -> cocitation_check
          -> llm_scoring -> update_corpus -> check_saturation
          -> (continue: select_seeds OR finalize: finalize -> END)
"""

from .types import DiffusionEngineState
from .graph import diffusion_engine_subgraph, create_diffusion_engine_subgraph
from .api import run_diffusion
from .citation_fetcher import fetch_citations_raw

__all__ = [
    "DiffusionEngineState",
    "diffusion_engine_subgraph",
    "create_diffusion_engine_subgraph",
    "run_diffusion",
    "fetch_citations_raw",
]
