"""Smart paper summary formatting with fallbacks for supervision loops.

Provides enhanced formatting that uses the best available data:
1. Detailed content from store (L2 10:1 summaries)
2. Structured fields (key_findings, methodology)
3. Short summary
4. Metadata fallback
"""

from typing import Optional

from workflows.research.academic_lit_review.state import PaperSummary


def format_paper_summary_enhanced(
    doi: str,
    summary: PaperSummary,
    detailed_content: Optional[str] = None,
    include_claims: bool = False,
    max_content_chars: int = 5000,
) -> str:
    """Format paper summary using best available data.

    Priority for content:
    1. detailed_content (L2 from store) if provided
    2. key_findings + methodology if substantial
    3. short_summary if not empty
    4. Title + metadata fallback

    Args:
        doi: Paper DOI
        summary: PaperSummary dict from workflow state
        detailed_content: L2 content from store (if fetched)
        include_claims: Whether to include extracted claims
        max_content_chars: Maximum characters for detailed content

    Returns:
        Formatted paper summary string
    """
    zotero_key = summary.get("zotero_key", doi.replace("/", "_")[:8])
    lines = [
        f"[@{zotero_key}] {summary.get('title', 'Untitled')} ({summary.get('year', 'N/A')})"
    ]

    # Authors
    authors = summary.get("authors", [])
    if authors:
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        lines.append(f"Authors: {author_str}")

    # Venue
    if summary.get("venue"):
        lines.append(f"Venue: {summary['venue']}")

    # Add detailed content if available (highest priority)
    if detailed_content:
        content = detailed_content[:max_content_chars]
        if len(detailed_content) > max_content_chars:
            content += "\n[... truncated ...]"
        lines.append(f"\nDetailed Content:\n{content}")
    else:
        # Fall back to structured fields
        has_substance = False

        key_findings = summary.get("key_findings", [])
        if key_findings:
            has_substance = True
            lines.append("\nKey Findings:")
            for i, finding in enumerate(key_findings[:5], 1):
                lines.append(f"  {i}. {finding}")

        methodology = summary.get("methodology", "")
        if methodology and methodology not in ("Not specified", "Not available", ""):
            has_substance = True
            lines.append(f"\nMethodology: {methodology}")

        short_summary = summary.get("short_summary", "")
        if short_summary:
            has_substance = True
            lines.append(f"\nSummary: {short_summary}")
        elif not has_substance:
            # Last resort: note that no detailed info is available
            lines.append("\n[No detailed summary available - content may be in store]")

        limitations = summary.get("limitations", [])
        if limitations:
            lines.append("\nLimitations:")
            for lim in limitations[:3]:
                lines.append(f"  - {lim}")

        if include_claims:
            claims = summary.get("claims", [])
            if claims:
                lines.append("\nKey Claims:")
                for claim in claims[:3]:
                    if isinstance(claim, dict):
                        lines.append(
                            f"  - {claim.get('claim', '')} "
                            f"(confidence: {claim.get('confidence', 0):.2f})"
                        )

    return "\n".join(lines)


def format_paper_summaries_with_budget(
    paper_summaries: dict[str, PaperSummary],
    detailed_content: dict[str, str],
    max_total_chars: int = 100000,
    papers_with_content_first: bool = True,
) -> str:
    """Format all paper summaries respecting character budget.

    Papers with detailed_content get full formatting first,
    remaining budget goes to structured fields.

    Args:
        paper_summaries: DOI -> PaperSummary mapping
        detailed_content: DOI -> L2 content mapping (from store)
        max_total_chars: Total character budget
        papers_with_content_first: Prioritize papers with detailed content

    Returns:
        Formatted string of all paper summaries within budget
    """
    formatted_papers: list[str] = []
    total_chars = 0
    processed_dois: set[str] = set()

    # First pass: papers with detailed content (most valuable)
    if papers_with_content_first:
        for doi, content in detailed_content.items():
            if doi not in paper_summaries:
                continue
            formatted = format_paper_summary_enhanced(
                doi, paper_summaries[doi], detailed_content=content
            )
            if total_chars + len(formatted) > max_total_chars:
                break
            formatted_papers.append(formatted)
            total_chars += len(formatted)
            processed_dois.add(doi)

    # Second pass: remaining papers without detailed content
    for doi, summary in paper_summaries.items():
        if doi in processed_dois:
            continue
        formatted = format_paper_summary_enhanced(doi, summary)
        if total_chars + len(formatted) > max_total_chars:
            # Add summary note about omitted papers
            remaining = (
                len(paper_summaries) - len(processed_dois) - len(formatted_papers)
            )
            if remaining > 0:
                formatted_papers.append(
                    f"\n[{remaining} additional papers available but omitted due to context limits]"
                )
            break
        formatted_papers.append(formatted)
        total_chars += len(formatted)
        processed_dois.add(doi)

    return "\n\n" + "=" * 60 + "\n\n".join(formatted_papers)


def create_manifest_note(
    papers_with_detail: int,
    papers_total: int,
    compression_level: int = 2,
) -> str:
    """Create a manifest note explaining data availability.

    Args:
        papers_with_detail: Number of papers with detailed content loaded
        papers_total: Total number of papers available
        compression_level: Which store level was queried

    Returns:
        Note string for inclusion in prompts
    """
    level_names = {0: "full documents", 1: "short summaries", 2: "10:1 summaries"}
    level_name = level_names.get(compression_level, f"L{compression_level}")

    return (
        f"[{papers_with_detail}/{papers_total} papers have {level_name} loaded. "
        f"Remaining papers show metadata and key findings only.]"
    )
