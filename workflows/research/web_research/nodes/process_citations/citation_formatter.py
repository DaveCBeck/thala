"""Citation text replacement utilities."""

import re


def _replace_citations_in_report(
    report: str,
    url_to_key: dict[str, str],
    citations: list[dict],
) -> str:
    """
    Replace numeric citations with Pandoc-style cite keys.

    Transforms:
    - Inline: [1] -> [@ABCD1234]
    - References section: [1] Title: URL -> [@ABCD1234] Title
    """
    # Build index-to-key mapping (same order as _extract_citations)
    index_to_key = {}
    seen_urls = set()
    idx = 1

    for citation in citations:
        url = citation.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            if url in url_to_key and url_to_key[url]:
                index_to_key[idx] = url_to_key[url]
            idx += 1

    if not index_to_key:
        return report

    # Replace inline citations [1], [2], etc. with [@KEY]
    def replace_inline(match):
        idx = int(match.group(1))
        if idx in index_to_key:
            return f"[@{index_to_key[idx]}]"
        return match.group(0)  # Keep original if no mapping

    updated_report = re.sub(r"\[(\d+)\]", replace_inline, report)

    # Update references section
    # Pattern: [N] Title: URL -> [@KEY] Title
    def replace_reference(match):
        idx = int(match.group(1))
        title = match.group(2)
        if idx in index_to_key:
            return f"[@{index_to_key[idx]}] {title}"
        return match.group(0)

    updated_report = re.sub(
        r"^\[(\d+)\]\s+(.+?):\s+https?://\S+\s*$",
        replace_reference,
        updated_report,
        flags=re.MULTILINE,
    )

    return updated_report
