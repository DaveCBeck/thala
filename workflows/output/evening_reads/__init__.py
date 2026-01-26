"""Evening Reads workflow.

Transforms academic literature reviews into a 4-part series:
1 overview + 3 deep-dives, each genuinely distinct and standalone.

Usage:
    from workflows.output.evening_reads import evening_reads_graph

    result = await evening_reads_graph.ainvoke({
        "input": {
            "literature_review": "Your literature review markdown here..."
        }
    })

    # Access the final outputs
    final_outputs = result["final_outputs"]
    for output in final_outputs:
        print(f"{output['id']}: {output['title']} ({output['word_count']} words)")

    # Check status
    if result["status"] == "success":
        print("All references resolved")
    elif result["status"] == "partial":
        print(f"Missing references: {result['missing_references']}")

    # Access planning details
    for assignment in result["deep_dive_assignments"]:
        print(f"{assignment['id']}: {assignment['title']}")
        print(f"  Theme: {assignment['theme']}")
        print(f"  Anchors: {assignment['anchor_keys']}")
"""

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

__all__ = [
    # Graph
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
