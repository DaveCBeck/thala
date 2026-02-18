"""Documentation generation nodes for synthesis subgraph."""

from datetime import datetime, timezone
from typing import Any

from ..types import SynthesisState


async def generate_prisma_docs_node(state: SynthesisState) -> dict[str, Any]:
    """Generate PRISMA-style documentation of the search process.

    Uses real counts from TransparencyReport when available,
    with fallback to basic counts from state.
    """
    input_data = state.get("input", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])
    report = state.get("transparency_report")

    topic = input_data.get("topic", "Unknown")
    total_papers = len(paper_summaries)

    if report:
        keyword_count = report.get("keyword_paper_count", 0)
        citation_count = report.get("citation_paper_count", 0)
        raw_results = report.get("raw_results_count", 0)
        total_rejected = report.get("total_rejected", 0)
        papers_failed = report.get("papers_failed_count", 0)
        metadata_only = report.get("metadata_only_count", 0)
        full_text_count = report.get("papers_processed_count", 0) - metadata_only

        prisma_doc = f"""# PRISMA Documentation

## Search Information

**Topic**: {topic}
**Date of Search**: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
**Database Searched**: OpenAlex

## Identification

- Records identified through OpenAlex keyword search: {raw_results}
- Records passing relevance threshold (>= {report.get("relevance_threshold", 0.6)}): {keyword_count}
- Additional records through citation network expansion: {citation_count}

## Screening

- Records screened for relevance: {raw_results}
- Records excluded (below relevance threshold): {raw_results - keyword_count}
- Records excluded during diffusion (not relevant): {total_rejected}

## Eligibility

- Papers assessed for full-text retrieval: {total_papers + papers_failed}
- Papers where full text was not retrievable: {papers_failed}

## Included

- Studies included in qualitative synthesis: {total_papers}
  - Full-text analysis: {full_text_count}
  - Metadata-only analysis: {metadata_only}
- Studies organized into thematic clusters: {len(clusters)}

## Thematic Distribution

"""
    else:
        # Backwards-compatible fallback without transparency report
        prisma_doc = f"""# PRISMA Documentation

## Search Information

**Topic**: {topic}
**Date of Search**: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
**Database Searched**: OpenAlex

## Included

- Studies included in qualitative synthesis: {total_papers}
- Studies organized into thematic clusters: {len(clusters)}

## Thematic Distribution

"""

    for cluster in clusters:
        prisma_doc += f"- {cluster['label']}: {len(cluster['paper_dois'])} papers\n"

    return {"prisma_documentation": prisma_doc}
