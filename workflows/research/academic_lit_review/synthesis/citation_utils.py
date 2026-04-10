"""Citation utilities for synthesis subgraph."""

import re
import unicodedata

from workflows.research.academic_lit_review.state import PaperSummary
from .types import QualityMetrics


# Characters that appear inside words as corrupted ligatures in PDF extraction.
# Bullet (U+2022) commonly replaces ff/fi/fl ligatures in Marker output and
# sometimes propagates through OpenAlex metadata scraped from publisher PDFs.
_MOJIBAKE_LIGATURE_PATTERN = re.compile(
    r"(?<=[a-zA-Z])\u2022(?=[a-zA-Z])"  # bullet between letters
)


def sanitize_metadata_text(text: str) -> str:
    """Clean common PDF/metadata encoding artifacts from text.

    Handles three classes of corruption that appear in OpenAlex metadata
    and Marker-extracted text:
    1. Unicode ligatures (ﬀ ﬁ ﬂ ﬃ ﬄ) — resolved by NFKC normalization
    2. Dotless-i (ı) and dotted-I (İ) — Turkish characters that replace
       ASCII i/I in corrupted metadata
    3. Bullet (•) replacing ligatures inside words — PDF extraction artifact
       where the ff/fi/fl glyph is misread as a bullet character
    """
    if not text:
        return text
    # NFKC: decomposes ligatures (ﬀ→ff, ﬁ→fi, ﬂ→fl) and recomposes
    text = unicodedata.normalize("NFKC", text)
    # Dotless-i → i, dotted-I → I (common in corrupted author names)
    text = text.replace("\u0131", "i").replace("\u0130", "I")
    # Word-internal bullet → ff (most common corrupted ligature by far;
    # covers "e•ects"→"effects", "di•erent"→"different", "o•er"→"offer")
    text = _MOJIBAKE_LIGATURE_PATTERN.sub("ff", text)
    return text


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

        title = sanitize_metadata_text(summary.get("title", "Unknown"))
        authors_str = ", ".join(
            sanitize_metadata_text(a) for a in summary.get("authors", [])[:3]
        )
        paper_text = f"""
[@{key}] {title} ({summary.get("year", "n.d.")})
  Authors: {authors_str}
  Key Findings: {"; ".join(summary.get("key_findings", [])[:2])}
  Methodology: {summary.get("methodology", "N/A")[:100]}"""

        formatted.append(paper_text)

    return "\n".join(formatted)


def extract_citations_from_text(text: str) -> list[str]:
    """Extract all [@KEY] citations from text."""
    pattern = r"\[@([^\]]+)\]"
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

    sections = re.split(r"^## ", review_text, flags=re.MULTILINE)
    sections_count = len(sections) - 1

    section_lengths = [len(s.split()) for s in sections[1:]] if sections_count > 0 else [0]
    avg_section_length = sum(section_lengths) // len(section_lengths) if section_lengths else 0

    # Issues are informational - actual pass/fail is determined by verify_quality_node
    # using quality-tier-aware thresholds. These thresholds should be lenient.
    issues = []
    if corpus_coverage < 0.3:  # Very low coverage - worth flagging
        issues.append(f"Low corpus coverage: only {corpus_coverage:.0%} of papers cited")
    if total_words < 1000:  # Very short - worth flagging
        issues.append(f"Review may be too short: {total_words} words")
    if citation_count < 5:  # Very few citations - worth flagging
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
