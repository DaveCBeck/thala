"""Loop 2: Literature Base Expansion subgraph.

Identifies missing literature bases and integrates mini-reviews to expand
the review's perspective beyond the initial corpus.
"""

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    QualitySettings,
)
from workflows.shared.llm_utils import ModelTier, get_structured_output

from ..types import LiteratureBase, LiteratureBaseDecision
from ..prompts import (
    LOOP2_ANALYZER_SYSTEM,
    LOOP2_ANALYZER_USER,
    LOOP2_INTEGRATOR_SYSTEM,
    LOOP2_INTEGRATOR_USER,
)
from ..mini_review import run_mini_review

logger = logging.getLogger(__name__)


# =============================================================================
# State Definition
# =============================================================================


class Loop2State(TypedDict, total=False):
    """State for Loop 2 literature base expansion."""

    current_review: str
    paper_corpus: dict
    paper_summaries: dict
    zotero_keys: dict
    input: LitReviewInput
    quality_settings: QualitySettings
    iteration: int
    max_iterations: int
    explored_bases: list[str]
    is_complete: bool
    decision: Optional[dict]
    # Error tracking fields
    errors: list[dict]
    iterations_failed: int
    consecutive_failures: int
    integration_failed: bool
    mini_review_failed: bool


# =============================================================================
# Nodes
# =============================================================================


async def analyze_for_bases_node(state: Loop2State) -> dict:
    """Analyze review to identify missing literature base."""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    logger.info(f"Loop 2 iteration {iteration + 1}/{max_iterations}: Analyzing for missing literature bases")

    input_data = state["input"]
    explored_bases_text = "\n".join(
        f"- {base}" for base in state.get("explored_bases", [])
    ) or "None yet"

    user_prompt = LOOP2_ANALYZER_USER.format(
        review=state["current_review"],
        topic=input_data["topic"],
        research_questions="\n".join(f"- {q}" for q in input_data["research_questions"]),
        explored_bases=explored_bases_text,
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        response = await get_structured_output(
            output_schema=LiteratureBaseDecision,
            user_prompt=user_prompt,
            system_prompt=LOOP2_ANALYZER_SYSTEM,
            tier=ModelTier.OPUS,
            max_tokens=2048,
            use_json_schema_method=True,
            max_retries=2,
        )

        logger.debug(f"Analyzer decision: {response.action}")
        if response.action == "expand_base":
            logger.info(f"Identified literature base: {response.literature_base.name}")

        return {"decision": response.model_dump()}

    except Exception as e:
        logger.error(f"Loop 2 analysis failed: {e}")
        errors = state.get("errors", [])
        return {
            "decision": {
                "action": "error",
                "literature_base": None,
                "reasoning": f"Analysis failed: {e}",
            },
            "errors": errors + [{
                "loop_number": 2,
                "iteration": iteration,
                "node_name": "analyze_for_bases",
                "error_type": "analysis_error",
                "error_message": str(e),
                "recoverable": True,
            }],
        }


async def run_mini_review_node(state: Loop2State) -> dict:
    """Execute mini-review on identified literature base."""
    decision = state.get("decision")
    errors = state.get("errors", [])

    if not decision:
        logger.warning("Mini-review node called without decision")
        return {
            "mini_review_failed": True,
            "errors": errors + [{
                "loop_number": 2,
                "iteration": state.get("iteration", 0),
                "node_name": "run_mini_review",
                "error_type": "validation_error",
                "error_message": "No decision provided",
                "recoverable": False,
            }],
        }

    if decision["action"] != "expand_base":
        logger.warning(f"Mini-review node called with invalid action: {decision['action']}")
        return {
            "mini_review_failed": True,
            "errors": errors + [{
                "loop_number": 2,
                "iteration": state.get("iteration", 0),
                "node_name": "run_mini_review",
                "error_type": "validation_error",
                "error_message": f"Expected expand_base action, got: {decision['action']}",
                "recoverable": False,
            }],
        }

    literature_base = LiteratureBase(**decision["literature_base"])
    logger.debug(f"Running mini-review for: {literature_base.name}")

    exclude_dois = set(state["paper_corpus"].keys())
    parent_topic = state["input"]["topic"]

    mini_review_result = await run_mini_review(
        literature_base=literature_base,
        parent_topic=parent_topic,
        quality_settings=state["quality_settings"],
        exclude_dois=exclude_dois,
    )

    new_paper_summaries = mini_review_result.get("paper_summaries", {})
    new_paper_corpus = mini_review_result.get("paper_corpus", {})
    new_zotero_keys = mini_review_result.get("zotero_keys", {})

    logger.info(
        f"Mini-review complete: {len(new_paper_summaries)} papers, "
        f"{len(mini_review_result.get('mini_review_text', ''))} chars"
    )

    return {
        "decision": {
            **decision,
            "mini_review_text": mini_review_result.get("mini_review_text", ""),
            "new_paper_summaries": new_paper_summaries,
            "new_paper_corpus": new_paper_corpus,  # Full PaperMetadata for merging
            "new_zotero_keys": new_zotero_keys,
            "clusters": mini_review_result.get("clusters", []),
            "references": mini_review_result.get("references", []),
        }
    }


async def integrate_findings_node(state: Loop2State) -> dict:
    """Integrate mini-review findings into main review."""
    from workflows.shared.llm_utils import get_llm

    iteration = state.get("iteration", 0)
    errors = state.get("errors", [])
    iterations_failed = state.get("iterations_failed", 0)

    try:
        decision = state.get("decision")
        if not decision or not decision.get("literature_base"):
            logger.error("Integration node called without valid decision")
            return {
                "errors": errors + [{
                    "loop_number": 2,
                    "iteration": iteration,
                    "node_name": "integrate_findings",
                    "error_type": "validation_error",
                    "error_message": "No valid decision with literature_base",
                    "recoverable": False,
                }],
                "iterations_failed": iterations_failed + 1,
                "integration_failed": True,
            }

        literature_base = LiteratureBase(**decision["literature_base"])

        logger.debug(f"Integrating findings for: {literature_base.name}")

        mini_review_text = decision.get("mini_review_text", "")
        new_paper_summaries = decision.get("new_paper_summaries", {})
        new_paper_corpus = decision.get("new_paper_corpus", {})
        new_zotero_keys = decision.get("new_zotero_keys", {})

        # Validate mini-review before consuming iteration
        if not mini_review_text or len(mini_review_text.strip()) < 100:
            logger.warning(f"Mini-review too short for base '{literature_base.name}' ({len(mini_review_text.strip())} chars)")
            return {
                # DON'T increment iteration on failure
                "errors": errors + [{
                    "loop_number": 2,
                    "iteration": iteration,
                    "node_name": "integrate_findings",
                    "error_type": "validation_error",
                    "error_message": f"Mini-review failed or too short ({len(mini_review_text.strip()) if mini_review_text else 0} chars)",
                    "recoverable": True,
                }],
                "iterations_failed": iterations_failed + 1,
                "integration_failed": True,
            }

        citation_keys = "\n".join(
            f"[@{key}] - {new_paper_summaries[doi]['title']}"
            for doi, key in new_zotero_keys.items()
            if doi in new_paper_summaries
        )

        user_prompt = LOOP2_INTEGRATOR_USER.format(
            current_review=state["current_review"],
            literature_base_name=literature_base.name,
            mini_review=mini_review_text,
            integration_strategy=literature_base.integration_strategy,
            new_citation_keys=citation_keys or "None",
        )

        llm = get_llm(ModelTier.OPUS, max_tokens=16384)
        response = await llm.ainvoke([
            {"role": "system", "content": LOOP2_INTEGRATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ])

        updated_review = response.content

        merged_summaries = {**state["paper_summaries"], **new_paper_summaries}
        merged_zotero = {**state["zotero_keys"], **new_zotero_keys}

        # Merge paper corpus with full PaperMetadata from mini-review
        merged_corpus = state["paper_corpus"].copy()
        new_papers_added = 0
        for doi, paper_metadata in new_paper_corpus.items():
            if doi not in merged_corpus:
                merged_corpus[doi] = paper_metadata
                new_papers_added += 1

        explored_bases = state.get("explored_bases", []) + [literature_base.name]

        logger.info(
            f"Integration complete: {len(updated_review)} chars, "
            f"{len(merged_summaries)} total papers, {new_papers_added} new papers"
        )

        return {
            "current_review": updated_review,
            "paper_summaries": merged_summaries,
            "zotero_keys": merged_zotero,
            "paper_corpus": merged_corpus,
            "explored_bases": explored_bases,
            "iteration": iteration + 1,  # Only increment on success
        }

    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return {
            "errors": errors + [{
                "loop_number": 2,
                "iteration": iteration,
                "node_name": "integrate_findings",
                "error_type": "integration_error",
                "error_message": str(e),
                "recoverable": True,
            }],
            "iterations_failed": iterations_failed + 1,
            "integration_failed": True,
        }


async def finalize_node(state: Loop2State) -> dict:
    """Mark loop as complete and return final state."""
    logger.debug("Loop 2 finalized")
    return {"is_complete": True}


# =============================================================================
# Routing
# =============================================================================


def route_after_analyze(state: Loop2State) -> str:
    """Route based on analyzer decision."""
    decision = state.get("decision")
    if not decision:
        return "finalize"

    # Handle error action
    if decision["action"] == "error":
        return "finalize"  # Will trigger check_continue for potential retry

    if decision["action"] == "expand_base":
        return "run_mini_review"
    return "finalize"


def check_continue(state: Loop2State) -> str:
    """Check if should continue iterating or complete."""
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # Check for failures that didn't increment iteration
    integration_failed = state.get("integration_failed", False)
    mini_review_failed = state.get("mini_review_failed", False)
    consecutive_failures = state.get("consecutive_failures", 0)

    if integration_failed or mini_review_failed:
        if consecutive_failures >= 2:
            logger.warning("Too many consecutive failures, completing Loop 2")
            return "finalize"
        # Allow retry
        return "analyze_for_bases"

    if iteration >= max_iterations:
        logger.debug(f"Max iterations ({max_iterations}) reached")
        return "finalize"

    return "analyze_for_bases"


# =============================================================================
# Graph Construction
# =============================================================================


def create_loop2_graph() -> StateGraph:
    """Create Loop 2 literature base expansion graph."""
    graph = StateGraph(Loop2State)

    graph.add_node("analyze_for_bases", analyze_for_bases_node)
    graph.add_node("run_mini_review", run_mini_review_node)
    graph.add_node("integrate_findings", integrate_findings_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "analyze_for_bases")
    graph.add_conditional_edges(
        "analyze_for_bases",
        route_after_analyze,
        {
            "run_mini_review": "run_mini_review",
            "finalize": "finalize",
        },
    )
    graph.add_edge("run_mini_review", "integrate_findings")
    graph.add_conditional_edges(
        "integrate_findings",
        check_continue,
        {
            "analyze_for_bases": "analyze_for_bases",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()


loop2_graph = create_loop2_graph()


# =============================================================================
# Standalone API
# =============================================================================


@traceable(run_type="chain", name="Loop2_LiteratureExpansion")
async def run_loop2_standalone(
    review: str,
    paper_corpus: dict,
    paper_summaries: dict,
    zotero_keys: dict,
    input_data: LitReviewInput,
    quality_settings: QualitySettings,
    max_iterations: int = 3,
    config: dict | None = None,
) -> dict:
    """Run Loop 2 as standalone operation for testing.

    Args:
        review: Current review text
        paper_corpus: DOI -> PaperMetadata mapping
        paper_summaries: DOI -> PaperSummary mapping
        zotero_keys: DOI -> Zotero key mapping
        input_data: LitReviewInput with topic and research questions
        quality_settings: Quality tier settings
        max_iterations: Maximum expansion iterations (default: 3)
        config: Optional LangGraph config with run_id and run_name for tracing

    Returns:
        Dict with:
            - current_review: Updated review text
            - paper_summaries: Merged paper summaries
            - zotero_keys: Merged Zotero keys
            - paper_corpus: Merged corpus
            - explored_bases: List of literature bases explored
            - iteration: Final iteration count
            - is_complete: Whether loop completed
    """
    initial_state = Loop2State(
        current_review=review,
        paper_corpus=paper_corpus,
        paper_summaries=paper_summaries,
        zotero_keys=zotero_keys,
        input=input_data,
        quality_settings=quality_settings,
        iteration=0,
        max_iterations=max_iterations,
        explored_bases=[],
        is_complete=False,
        decision=None,
        # Error tracking fields
        errors=[],
        iterations_failed=0,
        consecutive_failures=0,
        integration_failed=False,
        mini_review_failed=False,
    )

    if config:
        result = await loop2_graph.ainvoke(initial_state, config=config)
    else:
        result = await loop2_graph.ainvoke(initial_state)

    return {
        "current_review": result.get("current_review", review),
        "paper_summaries": result.get("paper_summaries", paper_summaries),
        "zotero_keys": result.get("zotero_keys", zotero_keys),
        "paper_corpus": result.get("paper_corpus", paper_corpus),
        "explored_bases": result.get("explored_bases", []),
        "iteration": result.get("iteration", 1),
        "is_complete": result.get("is_complete", False),
        # Error tracking fields
        "errors": result.get("errors", []),
        "iterations_failed": result.get("iterations_failed", 0),
    }
