"""Citation network subgraph for literature discovery via snowballing.

Fetches forward citations (papers that cite seeds) and backward citations
(papers cited by seeds) to expand the paper corpus through citation network
exploration.

Flow:
    START -> fetch_forward_citations -> fetch_backward_citations -> merge_and_filter -> END
"""

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from workflows.research.academic_lit_review.state import LitReviewInput
from workflows.research.academic_lit_review.quality_presets import QualitySettings

from .types import CitationNetworkState
from .traversal import fetch_forward_citations_node, fetch_backward_citations_node
from .scoring import merge_and_filter_node


def create_citation_network_subgraph() -> StateGraph:
    """Create the citation network discovery subgraph.

    Flow:
        START -> fetch_forward -> fetch_backward -> merge_and_filter -> END
    """
    builder = StateGraph(CitationNetworkState)

    builder.add_node("fetch_forward_citations", fetch_forward_citations_node)
    builder.add_node("fetch_backward_citations", fetch_backward_citations_node)
    builder.add_node("merge_and_filter", merge_and_filter_node)

    builder.add_edge(START, "fetch_forward_citations")
    builder.add_edge("fetch_forward_citations", "fetch_backward_citations")
    builder.add_edge("fetch_backward_citations", "merge_and_filter")
    builder.add_edge("merge_and_filter", END)

    return builder.compile()


citation_network_subgraph = create_citation_network_subgraph()


async def run_citation_expansion(
    seed_dois: list[str],
    topic: str,
    research_questions: list[str],
    quality_settings: QualitySettings,
    existing_dois: Optional[set[str]] = None,
    language_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Run citation network expansion as a standalone operation.

    Args:
        seed_dois: DOIs to expand citations from
        topic: Research topic
        research_questions: List of research questions
        quality_settings: Quality tier settings
        existing_dois: Optional set of DOIs already in corpus
        language_config: Optional language configuration for filtering

    Returns:
        Dict with discovered_papers, rejected_papers, discovered_dois, new_edges
    """
    input_data = LitReviewInput(
        topic=topic,
        research_questions=research_questions,
        quality="standard",
        date_range=None,
        language_code=language_config["code"] if language_config else "en",
    )

    initial_state = CitationNetworkState(
        input=input_data,
        quality_settings=quality_settings,
        seed_dois=seed_dois,
        existing_dois=existing_dois or set(),
        language_config=language_config,
        forward_results=[],
        backward_results=[],
        citation_edges=[],
        discovered_papers=[],
        rejected_papers=[],
        discovered_dois=[],
        new_edges=[],
    )

    result = await citation_network_subgraph.ainvoke(initial_state)
    return {
        "discovered_papers": result.get("discovered_papers", []),
        "rejected_papers": result.get("rejected_papers", []),
        "discovered_dois": result.get("discovered_dois", []),
        "new_edges": result.get("new_edges", []),
    }
