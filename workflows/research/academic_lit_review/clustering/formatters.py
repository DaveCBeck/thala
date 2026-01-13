"""Document formatting utilities for clustering."""

from workflows.research.academic_lit_review.state import PaperSummary


def prepare_document_for_clustering(summary: PaperSummary) -> str:
    """Format a paper summary for BERTopic clustering.

    Creates a text representation that emphasizes discriminative features
    to help differentiate papers within a focused research domain.

    Strategy:
    - De-emphasize common domain terms (these dominate embeddings)
    - Emphasize unique aspects: methodology, application domain, specific findings
    - Include structured metadata for better differentiation
    """
    parts = []

    # Title (but avoid it dominating - it's often very similar across papers)
    title = summary.get("title", "Untitled")
    parts.append(title)

    # Application domain / venue context (often discriminative)
    venue = summary.get("venue")
    if venue:
        parts.append(f"Published in: {venue}")

    # Methodology is often highly discriminative
    methodology = summary.get("methodology")
    if methodology:
        # Emphasize methodology by putting it early
        parts.append(f"Methodology: {methodology}")

    # Key findings - these are unique to each paper
    key_findings = summary.get("key_findings", [])
    if key_findings:
        # Use all findings for better discrimination
        parts.append("Key findings: " + "; ".join(key_findings))

    # Limitations and future work - often unique and discriminative
    limitations = summary.get("limitations", [])
    if limitations:
        parts.append("Limitations: " + "; ".join(limitations[:3]))

    future_work = summary.get("future_work", [])
    if future_work:
        parts.append("Future directions: " + "; ".join(future_work[:3]))

    # Themes - but only the more specific ones (skip first few generic ones)
    themes = summary.get("themes", [])
    if themes:
        # Skip potentially generic themes at the start, use later ones
        specific_themes = themes[2:] if len(themes) > 4 else themes
        if specific_themes:
            parts.append("Specific topics: " + ", ".join(specific_themes[:8]))

    # Short summary last (it's often dominated by generic domain language)
    if summary.get("short_summary"):
        # Truncate to reduce dominance of generic abstract language
        short_summary = summary["short_summary"][:300]
        parts.append(short_summary)

    return "\n".join(parts)


def format_paper_for_llm(doi: str, summary: PaperSummary) -> str:
    """Format a paper summary for LLM clustering prompt."""
    authors = summary.get("authors", [])
    authors_str = ", ".join(authors[:3])
    if len(authors) > 3:
        authors_str += " et al."

    key_findings = summary.get("key_findings", [])
    findings_str = "; ".join(key_findings[:3]) if key_findings else "Not extracted"

    return f"""DOI: {doi}
Title: {summary.get('title', 'Untitled')}
Authors: {authors_str}
Year: {summary.get('year', 'Unknown')}
Venue: {summary.get('venue', 'Unknown')}
Summary: {summary.get('short_summary', 'No summary available')[:500]}
Key Findings: {findings_str}
Methodology: {summary.get('methodology', 'Not specified')[:200]}
Themes: {', '.join(summary.get('themes', [])[:5])}"""
