"""Citation network analysis using networkx."""

import logging
from datetime import datetime
from typing import Optional

import networkx as nx

from workflows.research.subgraphs.academic_lit_review.state import (
    CitationEdge,
    PaperMetadata,
    PaperNode,
)

logger = logging.getLogger(__name__)


class CitationGraph:
    """Citation network for analyzing paper relationships."""

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

        self._invalidate_cache()

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

        self._invalidate_cache()
        return True

    def get_seminal_papers(self, top_n: int = 10) -> list[str]:
        """Get most-cited papers by in-degree, using global citation count as tiebreaker."""
        sorted_dois = sorted(
            self._nodes.keys(),
            key=lambda doi: (
                self._nodes[doi]["in_degree"],
                self._nodes[doi]["cited_by_count"],
            ),
            reverse=True,
        )
        return sorted_dois[:top_n]

    def get_bridging_papers(self, top_n: int = 10) -> list[str]:
        """Get papers with highest betweenness centrality (connecting different clusters)."""
        try:
            if self._cached_centrality is None:
                self._cached_centrality = nx.betweenness_centrality(self._graph)
                self._last_analysis_time = datetime.utcnow()

            sorted_dois = sorted(
                self._cached_centrality.keys(),
                key=lambda doi: self._cached_centrality[doi],
                reverse=True,
            )
            return sorted_dois[:top_n]
        except Exception as e:
            logger.error(f"Error computing betweenness centrality: {e}")
            return []

    def get_recent_impactful(
        self, years: int = 3, top_n: int = 10, current_year: Optional[int] = None
    ) -> list[str]:
        """Get recent papers with high citation velocity (citations per year)."""
        if current_year is None:
            current_year = datetime.utcnow().year

        cutoff_year = current_year - years
        recent_dois = []

        for doi, node in self._nodes.items():
            year = node["year"]
            if year >= cutoff_year:
                years_since_pub = max(current_year - year, 1)
                velocity = node["in_degree"] / years_since_pub
                recent_dois.append((doi, velocity))

        # Sort by velocity
        recent_dois.sort(key=lambda x: x[1], reverse=True)
        return [doi for doi, _ in recent_dois[:top_n]]

    def identify_clusters(self, algorithm: str = "louvain") -> list[list[str]]:
        """Identify clusters using community detection.

        Args:
            algorithm: "louvain" (requires python-louvain) or "label_propagation" (fallback)

        Returns:
            List of clusters, where each cluster is a list of DOIs
        """
        if len(self._nodes) < 2:
            return [[doi] for doi in self._nodes.keys()]

        # Convert to undirected for community detection
        undirected = self._graph.to_undirected()

        try:
            if algorithm == "louvain":
                try:
                    import community as community_louvain

                    partition = community_louvain.best_partition(undirected)
                except ImportError:
                    logger.warning(
                        "python-louvain not available, falling back to label propagation"
                    )
                    algorithm = "label_propagation"

            if algorithm == "label_propagation":
                from networkx.algorithms import community

                communities = community.label_propagation_communities(undirected)
                partition = {}
                for cluster_id, cluster_nodes in enumerate(communities):
                    for node in cluster_nodes:
                        partition[node] = cluster_id
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return [[doi] for doi in self._nodes.keys()]

        # Update node cluster assignments
        for doi, cluster_id in partition.items():
            if doi in self._nodes:
                self._nodes[doi]["cluster_id"] = cluster_id

        # Group DOIs by cluster
        clusters_dict: dict[int, list[str]] = {}
        for doi, cluster_id in partition.items():
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(doi)

        return list(clusters_dict.values())

    def get_cocitation_candidates(
        self, paper_doi: str, corpus_dois: set[str], threshold: int = 3
    ) -> bool:
        """Check if paper shares >= threshold citations with corpus.

        Checks both directions:
        - Papers this paper cites (backward)
        - Papers citing this paper (forward)
        """
        if paper_doi not in self._nodes:
            return False

        # Get papers cited by this paper (outgoing edges)
        cited_by_paper = set(self._graph.successors(paper_doi))

        # Get papers citing this paper (incoming edges)
        citing_paper = set(self._graph.predecessors(paper_doi))

        # Count overlaps with corpus
        backward_overlap = len(cited_by_paper & corpus_dois)
        forward_overlap = len(citing_paper & corpus_dois)

        total_overlap = backward_overlap + forward_overlap
        return total_overlap >= threshold

    def get_unexplored_citations(
        self, doi: str, explored_dois: set[str]
    ) -> tuple[list[str], list[str]]:
        """Get unexplored citations for a paper.

        Returns:
            Tuple of (unexplored_citing, unexplored_cited)
        """
        if doi not in self._nodes:
            return [], []

        # Papers citing this one (predecessors)
        citing = set(self._graph.predecessors(doi))
        unexplored_citing = [d for d in citing if d not in explored_dois]

        # Papers cited by this one (successors)
        cited = set(self._graph.successors(doi))
        unexplored_cited = [d for d in cited if d not in explored_dois]

        return unexplored_citing, unexplored_cited

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
    def from_serializable(cls, data: dict) -> "CitationGraph":
        """Reconstruct from serialized data."""
        graph = cls()

        # Add nodes directly (don't use add_paper to avoid re-processing)
        for node_data in data.get("nodes", []):
            doi = node_data["doi"]
            graph._nodes[doi] = PaperNode(
                doi=doi,
                title=node_data["title"],
                year=node_data["year"],
                cited_by_count=node_data["cited_by_count"],
                in_degree=0,  # Will be recalculated
                out_degree=0,  # Will be recalculated
                discovery_stage=node_data.get("discovery_stage", 0),
                cluster_id=node_data.get("cluster_id"),
            )
            graph._graph.add_node(doi)

        # Add edges (recalculates degrees)
        for edge_data in data.get("edges", []):
            graph.add_citation(
                citing_doi=edge_data["citing_doi"],
                cited_doi=edge_data["cited_doi"],
                edge_type=edge_data["edge_type"],
            )

        return graph

    def get_expansion_candidates(
        self, max_papers: int = 20, prioritize_recent: bool = True
    ) -> list[str]:
        """Get best papers to expand from in next diffusion stage.

        Combines scores from:
        - Seminal papers (high in-degree)
        - Bridging papers (high betweenness)
        - Recent impactful papers (high citation velocity)

        Returns ranked list of DOIs.
        """
        if len(self._nodes) == 0:
            return []

        # Get candidates from each strategy
        seminal = set(self.get_seminal_papers(top_n=max_papers))
        bridging = set(self.get_bridging_papers(top_n=max_papers))

        if prioritize_recent:
            recent = set(self.get_recent_impactful(top_n=max_papers))
        else:
            recent = set()

        # Combine with scores (higher score = appears in more lists)
        scores: dict[str, int] = {}
        for doi in seminal:
            scores[doi] = scores.get(doi, 0) + 3  # Weight seminal highest
        for doi in bridging:
            scores[doi] = scores.get(doi, 0) + 2
        for doi in recent:
            scores[doi] = scores.get(doi, 0) + 2

        # Sort by score
        ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        return ranked[:max_papers]

    def _invalidate_cache(self) -> None:
        """Invalidate cached analysis results."""
        self._cached_centrality = None
        self._last_analysis_time = None
