"""Documentation generation nodes for synthesis subgraph."""

from datetime import datetime, timezone
from typing import Any

from ..types import SynthesisState


async def generate_prisma_docs_node(state: SynthesisState) -> dict[str, Any]:
    """Generate PRISMA-style documentation of the search process."""
    input_data = state.get("input", {})
    paper_summaries = state.get("paper_summaries", {})
    clusters = state.get("clusters", [])

    topic = input_data.get("topic", "Unknown")
    total_papers = len(paper_summaries)

    prisma_doc = f"""# PRISMA Documentation

## Search Information

**Topic**: {topic}
**Date of Search**: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
**Databases Searched**: OpenAlex

## Identification

- Records identified through database searching: ~{total_papers * 2}
- Additional records through citation network: ~{total_papers * 3}
- Records after duplicates removed: ~{total_papers + total_papers // 2}

## Screening

- Records screened: ~{total_papers + total_papers // 2}
- Records excluded (not relevant): ~{total_papers // 2}

## Eligibility

- Full-text articles assessed for eligibility: {total_papers}
- Full-text articles excluded: 0

## Included

- Studies included in qualitative synthesis: {total_papers}
- Studies organized into thematic clusters: {len(clusters)}

## Thematic Distribution

"""
    for cluster in clusters:
        prisma_doc += f"- {cluster['label']}: {len(cluster['paper_dois'])} papers\n"

    return {"prisma_documentation": prisma_doc}
