"""Citation network discovery module."""

from .types import CitationNetworkState, MAX_CITATIONS_PER_PAPER, MAX_CONCURRENT_FETCHES
from .traversal import fetch_forward_citations_node, fetch_backward_citations_node
from .scoring import merge_and_filter_node
from .core import (
    create_citation_network_subgraph,
    citation_network_subgraph,
    run_citation_expansion,
)

__all__ = [
    # Types
    "CitationNetworkState",
    "MAX_CITATIONS_PER_PAPER",
    "MAX_CONCURRENT_FETCHES",
    # Traversal
    "fetch_forward_citations_node",
    "fetch_backward_citations_node",
    # Scoring
    "merge_and_filter_node",
    # Core
    "create_citation_network_subgraph",
    "citation_network_subgraph",
    "run_citation_expansion",
]
