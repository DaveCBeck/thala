"""Citation network analysis using networkx."""

from workflows.academic_lit_review.citation_graph.algorithms import (
    CitationGraphAlgorithms,
)
from workflows.academic_lit_review.citation_graph.builder import CitationGraphBuilder
from workflows.academic_lit_review.citation_graph.types import CitationEdge, PaperNode
from workflows.academic_lit_review.state import PaperMetadata


class CitationGraph:
    """Citation network for analyzing paper relationships."""

    def __init__(self):
        self._builder = CitationGraphBuilder()
        self._algorithms = CitationGraphAlgorithms(self._builder)

    @property
    def node_count(self) -> int:
        return self._builder.node_count

    @property
    def edge_count(self) -> int:
        return self._builder.edge_count

    def add_paper(self, doi: str, metadata: PaperMetadata) -> None:
        """Add or update paper node."""
        self._builder.add_paper(doi, metadata)

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
        return self._builder.add_citation(citing_doi, cited_doi, edge_type)

    def get_seminal_papers(self, top_n: int = 10) -> list[str]:
        """Get most-cited papers by in-degree, using global citation count as tiebreaker."""
        return self._algorithms.get_seminal_papers(top_n)

    def get_bridging_papers(self, top_n: int = 10) -> list[str]:
        """Get papers with highest betweenness centrality (connecting different clusters)."""
        return self._algorithms.get_bridging_papers(top_n)

    def get_recent_impactful(
        self, years: int = 3, top_n: int = 10, current_year: int | None = None
    ) -> list[str]:
        """Get recent papers with high citation velocity (citations per year)."""
        return self._algorithms.get_recent_impactful(years, top_n, current_year)

    def identify_clusters(self, algorithm: str = "louvain") -> list[list[str]]:
        """Identify clusters using community detection.

        Args:
            algorithm: "louvain" (requires python-louvain) or "label_propagation" (fallback)

        Returns:
            List of clusters, where each cluster is a list of DOIs
        """
        return self._algorithms.identify_clusters(algorithm)

    def get_cocitation_candidates(
        self, paper_doi: str, corpus_dois: set[str], threshold: int = 3
    ) -> bool:
        """Check if paper shares >= threshold citations with corpus.

        Checks both directions:
        - Papers this paper cites (backward)
        - Papers citing this paper (forward)
        """
        return self._algorithms.get_cocitation_candidates(
            paper_doi, corpus_dois, threshold
        )

    def get_unexplored_citations(
        self, doi: str, explored_dois: set[str]
    ) -> tuple[list[str], list[str]]:
        """Get unexplored citations for a paper.

        Returns:
            Tuple of (unexplored_citing, unexplored_cited)
        """
        return self._algorithms.get_unexplored_citations(doi, explored_dois)

    def to_serializable(self) -> dict:
        """Convert to serializable format."""
        return self._builder.to_serializable()

    @classmethod
    def from_serializable(cls, data: dict) -> "CitationGraph":
        """Reconstruct from serialized data."""
        graph = cls()
        graph._builder = CitationGraphBuilder.from_serializable(data)
        graph._algorithms = CitationGraphAlgorithms(graph._builder)
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
        return self._algorithms.get_expansion_candidates(max_papers, prioritize_recent)


__all__ = [
    "CitationGraph",
    "CitationGraphBuilder",
    "CitationGraphAlgorithms",
    "CitationEdge",
    "PaperNode",
]
