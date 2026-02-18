"""Editorial transparency: summarise significant editing changes.

Rule-based filtering over structured enhancement data — no LLM calls.
Produces a brief prose summary suitable for appending to the methodology
section after the editing phase completes.
"""

# Significance thresholds — pinned as constants
SIGNIFICANT_WORD_DELTA_PCT = 0.20
LOW_CONFIDENCE_THRESHOLD = 0.7
MIN_CITATION_CHANGES = 1  # Any citation change is significant


def summarise_editorial_changes(section_enhancements: list[dict]) -> str:
    """Summarise significant editorial changes for methodology transparency.

    Filters enhancement results for significant changes (not cosmetic) and
    returns a brief 2-4 sentence summary. Returns empty string if no
    significant changes occurred.

    Args:
        section_enhancements: List of enhancement result dicts from
            EditingState.section_enhancements. Each contains:
            - citations_added (list[str])
            - citations_removed (list[str])
            - confidence (float)
            - original_word_count (int)
            - enhanced_word_count (int)
            - success (bool)
    """
    if not section_enhancements:
        return ""

    successful = [e for e in section_enhancements if e.get("success", False)]
    if not successful:
        return ""

    # Classify significant changes
    total_citations_added = 0
    total_citations_removed = 0
    sections_with_citation_changes = 0
    low_confidence_sections = 0
    large_word_delta_sections = 0
    total_original_words = 0
    total_enhanced_words = 0

    for enhancement in successful:
        added = len(enhancement.get("citations_added", []))
        removed = len(enhancement.get("citations_removed", []))
        confidence = enhancement.get("confidence", 1.0)
        original_wc = enhancement.get("original_word_count", 0)
        enhanced_wc = enhancement.get("enhanced_word_count", 0)

        total_citations_added += added
        total_citations_removed += removed
        total_original_words += original_wc
        total_enhanced_words += enhanced_wc

        if added >= MIN_CITATION_CHANGES or removed >= MIN_CITATION_CHANGES:
            sections_with_citation_changes += 1

        if confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence_sections += 1

        if original_wc > 0:
            delta = abs(enhanced_wc - original_wc) / original_wc
            if delta >= SIGNIFICANT_WORD_DELTA_PCT:
                large_word_delta_sections += 1

    # Only report if there are significant changes
    has_significant = sections_with_citation_changes > 0 or low_confidence_sections > 0 or large_word_delta_sections > 0
    if not has_significant:
        return ""

    # Build summary sentences
    sentences = ["Following initial synthesis, the review underwent automated editorial enhancement."]

    if total_citations_added > 0 or total_citations_removed > 0:
        citation_parts = []
        if total_citations_added > 0:
            citation_parts.append(
                f"the addition of {total_citations_added} citations "
                f"across {sections_with_citation_changes} sections to strengthen evidential support"
            )
        if total_citations_removed > 0:
            citation_parts.append(f"the removal of {total_citations_removed} citations")
        sentences.append(f"Significant changes included {' and '.join(citation_parts)}.")

    if low_confidence_sections > 0:
        sentences.append(
            f"{low_confidence_sections} section{'s were' if low_confidence_sections > 1 else ' was'} "
            f"restructured where initial synthesis confidence was below threshold."
        )

    if total_original_words > 0:
        overall_delta = (total_enhanced_words - total_original_words) / total_original_words
        sentences.append(f"Total word count changed by approximately {abs(overall_delta):.0%} during editing.")

    return " ".join(sentences)
