"""Helper functions for relevance scoring strategies."""

from workflows.research.academic_lit_review.state import PaperMetadata


def format_paper_for_batch(paper: PaperMetadata) -> str:
    """Format a single paper for inclusion in batch relevance scoring prompt."""
    authors_str = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5])
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    return f"""---
DOI: {paper.get("doi", "Unknown")}
Title: {paper.get("title", "Unknown")}
Authors: {authors_str or "Unknown"}
Year: {paper.get("year", "Unknown")}
Venue: {paper.get("venue", "Unknown")}
Abstract: {(paper.get("abstract") or "No abstract available")[:1000]}
Primary Topic: {paper.get("primary_topic", "Not specified")}"""


def chunk_papers(
    papers: list[PaperMetadata], chunk_size: int = 10
) -> list[list[PaperMetadata]]:
    """Split papers into chunks for batch processing."""
    return [papers[i : i + chunk_size] for i in range(0, len(papers), chunk_size)]
