"""Substack Review workflow.

Transforms academic literature reviews into polished Substack-style essays
through parallel generation and structured selection.

Usage:
    from workflows.output.substack_review import substack_review_graph

    result = await substack_review_graph.ainvoke({
        "input": {
            "literature_review": "Your literature review markdown here..."
        }
    })

    # Access the final essay
    final_essay = result["final_essay"]

    # Check status
    if result["status"] == "success":
        print("All references resolved")
    elif result["status"] == "partial":
        print(f"Missing references: {result['missing_references']}")

    # Access selection reasoning
    print(f"Selected angle: {result['selected_angle']}")
    print(f"Reasoning: {result['selection_reasoning']}")
"""

from .graph import substack_review_graph, create_substack_review_graph
from .state import SubstackReviewState, EssayInput, EssayDraft
from .schemas import ChoosingAgentOutput, EssayEvaluation

__all__ = [
    # Graph
    "substack_review_graph",
    "create_substack_review_graph",
    # State types
    "SubstackReviewState",
    "EssayInput",
    "EssayDraft",
    # Schemas
    "ChoosingAgentOutput",
    "EssayEvaluation",
]
