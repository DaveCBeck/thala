"""Citation graph analysis algorithms."""

import logging
from datetime import datetime, timezone
from typing import Optional

import networkx as nx

from workflows.research.academic_lit_review.citation_graph.builder import (
    CitationGraphBuilder,
)

logger = logging.getLogger(__name__)


class CitationGraphAlgorithms:
    """Graph analysis algorithms for citation networks."""

    def __init__(self, builder: CitationGraphBuilder):
        self._builder = builder

    def get_seminal_papers(self, top_n: int = 10) -> list[str]:
        """Get most-cited papers by in-degree, using global citation count as tiebreaker."""
        sorted_dois = sorted(
            self._builder.nodes.keys(),
            key=lambda doi: (
                self._builder.nodes[doi]["in_degree"],
                self._builder.nodes[doi]["cited_by_count"],
            ),
            reverse=True,
        )
        return sorted_dois[:top_n]

    def get_bridging_papers(self, top_n: int = 10) -> list[str]:
        """Get papers with highest betweenness centrality (connecting different clusters)."""
        try:
            if self._builder.cached_centrality is None:
                centrality = nx.betweenness_centrality(self._builder.graph)
                self._builder.cached_centrality = centrality

            sorted_dois = sorted(
                self._builder.cached_centrality.keys(),
                key=lambda doi: self._builder.cached_centrality[doi],
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
            current_year = datetime.now(timezone.utc).year

        cutoff_year = current_year - years
        recent_dois = []

        for doi, node in self._builder.nodes.items():
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
        if len(self._builder.nodes) < 2:
            return [[doi] for doi in self._builder.nodes.keys()]

        # Convert to undirected for community detection
        undirected = self._builder.graph.to_undirected()

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
            return [[doi] for doi in self._builder.nodes.keys()]

        # Update node cluster assignments
        for doi, cluster_id in partition.items():
            if doi in self._builder.nodes:
                self._builder.nodes[doi]["cluster_id"] = cluster_id

        # Group DOIs by cluster
        clusters_dict: dict[int, list[str]] = {}
        for doi, cluster_id in partition.items():
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(doi)

        return list(clusters_dict.values())

    def get_corpus_overlap_count(
        self, paper_doi: str, corpus_dois: set[str]
    ) -> int:
        """Count how many corpus papers this paper shares citations with.

        Checks both directions:
        - Papers this paper cites that are in corpus (backward overlap)
        - Papers citing this paper that are in corpus (forward overlap)

        Returns:
            Total count of corpus papers connected to this paper.
        """
        if paper_doi not in self._builder.nodes:
            return 0

        # Get papers cited by this paper (outgoing edges)
        cited_by_paper = set(self._builder.graph.successors(paper_doi))

        # Get papers citing this paper (incoming edges)
        citing_paper = set(self._builder.graph.predecessors(paper_doi))

        # Count overlaps with corpus
        backward_overlap = len(cited_by_paper & corpus_dois)
        forward_overlap = len(citing_paper & corpus_dois)

        return backward_overlap + forward_overlap

    def get_cocitation_candidates(
        self, paper_doi: str, corpus_dois: set[str], threshold: int = 3
    ) -> bool:
        """Check if paper shares >= threshold citations with corpus.

        Checks both directions:
        - Papers this paper cites (backward)
        - Papers citing this paper (forward)
        """
        return self.get_corpus_overlap_count(paper_doi, corpus_dois) >= threshold

    def get_unexplored_citations(
        self, doi: str, explored_dois: set[str]
    ) -> tuple[list[str], list[str]]:
        """Get unexplored citations for a paper.

        Returns:
            Tuple of (unexplored_citing, unexplored_cited)
        """
        if doi not in self._builder.nodes:
            return [], []

        # Papers citing this one (predecessors)
        citing = set(self._builder.graph.predecessors(doi))
        unexplored_citing = [d for d in citing if d not in explored_dois]

        # Papers cited by this one (successors)
        cited = set(self._builder.graph.successors(doi))
        unexplored_cited = [d for d in cited if d not in explored_dois]

        return unexplored_citing, unexplored_cited

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
        if len(self._builder.nodes) == 0:
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
