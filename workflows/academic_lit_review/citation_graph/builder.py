"""Citation graph construction and serialization."""

import logging
from datetime import datetime
from typing import Optional

import networkx as nx

from workflows.academic_lit_review.state import (
    CitationEdge,
    PaperMetadata,
    PaperNode,
)

logger = logging.getLogger(__name__)


class CitationGraphBuilder:
    """Manages citation graph construction and serialization."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, PaperNode] = {}
        self._edges: list[CitationEdge] = []
        self._last_analysis_time: Optional[datetime] = None
        self._cached_centrality: Optional[dict[str, float]] = None

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def graph(self) -> nx.DiGraph:
        """Access to underlying networkx graph."""
        return self._graph

    @property
    def nodes(self) -> dict[str, PaperNode]:
        """Access to node dictionary."""
        return self._nodes

    @property
    def edges(self) -> list[CitationEdge]:
        """Access to edge list."""
        return self._edges

    @property
    def cached_centrality(self) -> Optional[dict[str, float]]:
        """Access to cached centrality scores."""
        return self._cached_centrality

    @cached_centrality.setter
    def cached_centrality(self, value: Optional[dict[str, float]]) -> None:
        """Set cached centrality scores."""
        self._cached_centrality = value
        if value is not None:
            self._last_analysis_time = datetime.utcnow()

    def add_paper(self, doi: str, metadata: PaperMetadata) -> None:
        """Add or update paper node."""
        if doi in self._nodes:
            # Update fields but preserve degrees
            self._nodes[doi]["title"] = metadata.get("title", "")
            self._nodes[doi]["year"] = metadata.get("year", 0)
            self._nodes[doi]["cited_by_count"] = metadata.get("cited_by_count", 0)
        else:
            # Create new node from metadata
            self._nodes[doi] = PaperNode(
                doi=doi,
                title=metadata.get("title", ""),
                year=metadata.get("year", 0),
                cited_by_count=metadata.get("cited_by_count", 0),
                in_degree=0,
                out_degree=0,
                discovery_stage=metadata.get("discovery_stage", 0),
                cluster_id=None,
            )
            self._graph.add_node(doi)

        self.invalidate_cache()

    def add_citation(
        self, citing_doi: str, cited_doi: str, edge_type: str = "forward"
    ) -> bool:
        """Add directed edge (citing -> cited).

        Args:
            citing_doi: DOI of paper that cites
            cited_doi: DOI of paper being cited
            edge_type: "forward" or "backward"

        Returns:
            False if edge already exists, True if added
        """
        # Ensure both nodes exist
        if citing_doi not in self._nodes or cited_doi not in self._nodes:
            logger.warning(
                f"Cannot add citation: missing node(s) {citing_doi}, {cited_doi}"
            )
            return False

        # Check if edge exists
        if self._graph.has_edge(citing_doi, cited_doi):
            return False

        # Add edge
        self._graph.add_edge(citing_doi, cited_doi)
        self._edges.append(
            CitationEdge(
                citing_doi=citing_doi,
                cited_doi=cited_doi,
                edge_type=edge_type,
            )
        )

        # Update degrees
        self._nodes[citing_doi]["out_degree"] += 1
        self._nodes[cited_doi]["in_degree"] += 1

        self.invalidate_cache()
        return True

    def to_serializable(self) -> dict:
        """Convert to serializable format."""
        return {
            "nodes": [
                {
                    "doi": node["doi"],
                    "title": node["title"],
                    "year": node["year"],
                    "cited_by_count": node["cited_by_count"],
                    "in_degree": node["in_degree"],
                    "out_degree": node["out_degree"],
                    "discovery_stage": node["discovery_stage"],
                    "cluster_id": node["cluster_id"],
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "citing_doi": edge["citing_doi"],
                    "cited_doi": edge["cited_doi"],
                    "edge_type": edge["edge_type"],
                }
                for edge in self._edges
            ],
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "CitationGraphBuilder":
        """Reconstruct from serialized data."""
        builder = cls()

        # Add nodes directly (don't use add_paper to avoid re-processing)
        for node_data in data.get("nodes", []):
            doi = node_data["doi"]
            builder._nodes[doi] = PaperNode(
                doi=doi,
                title=node_data["title"],
                year=node_data["year"],
                cited_by_count=node_data["cited_by_count"],
                in_degree=0,  # Will be recalculated
                out_degree=0,  # Will be recalculated
                discovery_stage=node_data.get("discovery_stage", 0),
                cluster_id=node_data.get("cluster_id"),
            )
            builder._graph.add_node(doi)

        # Add edges (recalculates degrees)
        for edge_data in data.get("edges", []):
            builder.add_citation(
                citing_doi=edge_data["citing_doi"],
                cited_doi=edge_data["cited_doi"],
                edge_type=edge_data["edge_type"],
            )

        return builder

    def invalidate_cache(self) -> None:
        """Invalidate cached analysis results."""
        self._cached_centrality = None
        self._last_analysis_time = None
