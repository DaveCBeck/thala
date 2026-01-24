"""Helper functions for relevance scoring strategies."""

from workflows.research.academic_lit_review.state import PaperMetadata


def format_paper_for_batch(paper: PaperMetadata) -> str:
    """Format a single paper for inclusion in batch relevance scoring prompt.

    Includes corpus_cocitations if available (set during diffusion engine filtering).
    """
    authors_str = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:5])
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    # Build base fields
    lines = [
        "---",
        f"DOI: {paper.get('doi', 'Unknown')}",
        f"Title: {paper.get('title', 'Unknown')}",
        f"Authors: {authors_str or 'Unknown'}",
        f"Year: {paper.get('year', 'Unknown')}",
        f"Venue: {paper.get('venue', 'Unknown')}",
        f"Abstract: {(paper.get('abstract') or 'No abstract available')[:1000]}",
        f"Primary Topic: {paper.get('primary_topic', 'Not specified')}",
    ]

    # Add corpus co-citations if available
    corpus_cocitations = paper.get("corpus_cocitations")
    if corpus_cocitations is not None:
        lines.append(f"Corpus Co-citations: {corpus_cocitations}")

    return "\n".join(lines)


def chunk_papers(
    papers: list[PaperMetadata], chunk_size: int = 10
) -> list[list[PaperMetadata]]:
    """Split papers into chunks for batch processing."""
    return [papers[i : i + chunk_size] for i in range(0, len(papers), chunk_size)]
