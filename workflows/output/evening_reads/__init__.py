"""Evening Reads workflow.

Transforms academic literature reviews into a 4-part series:
1 overview + 3 deep-dives, each genuinely distinct and standalone.

Usage:
    from workflows.output.evening_reads import evening_reads

    result = await evening_reads(
        literature_review="Your literature review markdown here...",
    )

    # Access the final outputs
    final_outputs = result["final_outputs"]
    for output in final_outputs:
        print(f"{output['id']}: {output['title']} ({output['word_count']} words)")
"""

import re

from langsmith import traceable

from core.task_queue.task_context import get_trace_metadata, get_trace_tags

from .graph import evening_reads_graph, create_evening_reads_graph
from .state import (
    EveningReadsState,
    EveningReadsInput,
    CitationKeyMapping,
    DeepDiveAssignment,
    EnrichedContent,
    DeepDiveDraft,
    OverviewDraft,
    FormattedReference,
    FinalOutput,
)
from .schemas import PlanningOutput, DeepDiveTopicPlan


def _extract_topic(literature_review: str) -> str:
    """Extract topic from the first heading line of a literature review."""
    for line in literature_review.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return "Untitled"


@traceable(run_type="chain", name="EveningReads")
async def evening_reads(
    literature_review: str,
    editorial_stance: str = "",
) -> dict:
    """Run the evening reads workflow with full tracing.

    Args:
        literature_review: Literature review markdown to transform
        editorial_stance: Optional editorial stance to guide tone

    Returns:
        Evening reads state dict with final_outputs, status, etc.
    """
    topic = _extract_topic(literature_review)

    result = await evening_reads_graph.ainvoke(
        {
            "input": {
                "literature_review": literature_review,
                "editorial_stance": editorial_stance,
            }
        },
        config={
            "run_name": f"evening_reads:{topic[:60]}",
            "tags": [
                "workflow:evening_reads",
                *get_trace_tags(),
            ],
            "metadata": {
                "topic": topic[:100],
                **get_trace_metadata(),
            },
        },
    )

    return result


__all__ = [
    # Main API
    "evening_reads",
    # Graph (for direct access if needed)
    "evening_reads_graph",
    "create_evening_reads_graph",
    # State types
    "EveningReadsState",
    "EveningReadsInput",
    "CitationKeyMapping",
    "DeepDiveAssignment",
    "EnrichedContent",
    "DeepDiveDraft",
    "OverviewDraft",
    "FormattedReference",
    "FinalOutput",
    # Schemas
    "PlanningOutput",
    "DeepDiveTopicPlan",
]
