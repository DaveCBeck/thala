"""Graph definition for synthesis subgraph."""

from langgraph.graph import END, START, StateGraph

from .types import SynthesisState
from .nodes import (
    write_intro_methodology_node,
    write_thematic_sections_node,
    write_discussion_conclusions_node,
    integrate_sections_node,
    process_citations_node,
    verify_quality_node,
    generate_prisma_docs_node,
)


def create_synthesis_subgraph() -> StateGraph:
    """Create the synthesis/writing subgraph.

    Flow:
        START -> write_intro_methodology -> write_thematic_sections
              -> write_discussion_conclusions -> integrate_sections
              -> process_citations -> verify_quality -> prisma_docs -> END
    """
    builder = StateGraph(SynthesisState)

    builder.add_node("write_intro_methodology", write_intro_methodology_node)
    builder.add_node("write_thematic_sections", write_thematic_sections_node)
    builder.add_node("write_discussion_conclusions", write_discussion_conclusions_node)
    builder.add_node("integrate_sections", integrate_sections_node)
    builder.add_node("process_citations", process_citations_node)
    builder.add_node("verify_quality", verify_quality_node)
    builder.add_node("generate_prisma_docs", generate_prisma_docs_node)

    builder.add_edge(START, "write_intro_methodology")
    builder.add_edge("write_intro_methodology", "write_thematic_sections")
    builder.add_edge("write_thematic_sections", "write_discussion_conclusions")
    builder.add_edge("write_discussion_conclusions", "integrate_sections")
    builder.add_edge("integrate_sections", "process_citations")
    builder.add_edge("process_citations", "verify_quality")
    builder.add_edge("verify_quality", "generate_prisma_docs")
    builder.add_edge("generate_prisma_docs", END)

    return builder.compile()


synthesis_subgraph = create_synthesis_subgraph()
