"""Citation processing nodes for synthesis subgraph."""

import logging
from typing import Any

from workflows.academic_lit_review.state import FormattedCitation
from ..types import SynthesisState
from ..citation_utils import extract_citations_from_text

logger = logging.getLogger(__name__)


async def process_citations_node(state: SynthesisState) -> dict[str, Any]:
    """Process citations and build reference list."""
    integrated_review = state.get("integrated_review", "")
    paper_summaries = state.get("paper_summaries", {})
    zotero_keys = state.get("zotero_keys", {})

    citation_keys_used = extract_citations_from_text(integrated_review)

    references: list[FormattedCitation] = []
    key_to_doi = {v: k for k, v in zotero_keys.items()}

    for key in sorted(citation_keys_used):
        doi = key_to_doi.get(key)
        if doi and doi in paper_summaries:
            summary = paper_summaries[doi]
            authors = summary.get("authors", [])
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."

            citation_text = (
                f"{authors_str} ({summary.get('year', 'n.d.')}). "
                f"{summary.get('title', 'Untitled')}. "
                f"{summary.get('venue', 'Unknown venue')}."
            )

            references.append(FormattedCitation(
                doi=doi,
                citation_text=citation_text,
                zotero_key=key,
            ))

    if references:
        references_text = "\n\n## References\n\n"
        for ref in references:
            references_text += f"[@{ref['zotero_key']}] {ref['citation_text']}\n\n"

        final_review = integrated_review + references_text
    else:
        final_review = integrated_review

    logger.info(f"Processed {len(references)} citations")

    return {
        "final_review": final_review,
        "references": references,
        "citation_keys": citation_keys_used,
    }
