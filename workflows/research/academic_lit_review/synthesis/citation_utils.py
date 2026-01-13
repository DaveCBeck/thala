"""Citation utilities for synthesis subgraph."""

import re

from workflows.research.academic_lit_review.state import PaperSummary
from .types import QualityMetrics


def format_papers_with_keys(
    dois: list[str],
    paper_summaries: dict[str, PaperSummary],
    zotero_keys: dict[str, str],
) -> str:
    """Format papers with their Zotero citation keys for the prompt.

    Raises:
        ValueError: If any paper is missing a Zotero key. All papers must have
            real Zotero keys from document_processing.
    """
    formatted = []

    for doi in dois:
        summary = paper_summaries.get(doi)
        if not summary:
            continue

        key = zotero_keys.get(doi)
        if not key:
            raise ValueError(
                f"Paper {doi} ({summary.get('title', 'Unknown')}) has no Zotero key. "
                f"Document processing may have failed for this paper. "
                f"Check logs for processing errors."
            )

        paper_text = f"""
[@{key}] {summary.get('title', 'Unknown')} ({summary.get('year', 'n.d.')})
  Authors: {', '.join(summary.get('authors', [])[:3])}
  Key Findings: {'; '.join(summary.get('key_findings', [])[:2])}
  Methodology: {summary.get('methodology', 'N/A')[:100]}"""

        formatted.append(paper_text)

    return "\n".join(formatted)


def extract_citations_from_text(text: str) -> list[str]:
    """Extract all [@KEY] citations from text."""
    pattern = r'\[@([^\]]+)\]'
    matches = re.findall(pattern, text)

    keys = []
    for match in matches:
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.append(key)

    return list(set(keys))


def calculate_quality_metrics(
    review_text: str,
    paper_summaries: dict[str, PaperSummary],
    zotero_keys: dict[str, str],
) -> QualityMetrics:
    """Calculate quality metrics for the review."""
    words = review_text.split()
    total_words = len(words)

    citation_keys = extract_citations_from_text(review_text)
    citation_count = len(citation_keys)

    key_to_doi = {v: k for k, v in zotero_keys.items()}
    cited_dois = set()
    for key in citation_keys:
        if key in key_to_doi:
            cited_dois.add(key_to_doi[key])

    unique_papers_cited = len(cited_dois)
    corpus_size = len(paper_summaries)
    corpus_coverage = unique_papers_cited / corpus_size if corpus_size > 0 else 0

    all_dois = set(paper_summaries.keys())
    uncited_papers = list(all_dois - cited_dois)

    sections = re.split(r'^## ', review_text, flags=re.MULTILINE)
    sections_count = len(sections) - 1

    section_lengths = [len(s.split()) for s in sections[1:]] if sections_count > 0 else [0]
    avg_section_length = sum(section_lengths) // len(section_lengths) if section_lengths else 0

    issues = []
    if corpus_coverage < 0.5:
        issues.append(f"Low corpus coverage: only {corpus_coverage:.0%} of papers cited")
    if total_words < 5000:
        issues.append(f"Review may be too short: {total_words} words")
    if citation_count < 20:
        issues.append(f"Low citation count: {citation_count} citations")

    return QualityMetrics(
        total_words=total_words,
        citation_count=citation_count,
        unique_papers_cited=unique_papers_cited,
        corpus_coverage=corpus_coverage,
        uncited_papers=uncited_papers[:20],
        sections_count=sections_count,
        avg_section_length=avg_section_length,
        issues=issues,
    )
