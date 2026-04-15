"""Loop 2: Literature Base Expansion subgraph.

Identifies missing literature bases and integrates mini-reviews to expand
the review's perspective beyond the initial corpus.

This is a standalone copy for the enhancement workflow, importing types,
prompts, and mini_review from the original supervised_lit_review location.
"""

import logging
from operator import add
from typing import Annotated, Any, Optional

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from core.task_queue.schemas import IncrementalCheckpointCallback

from workflows.research.academic_lit_review.state import (
    LitReviewInput,
    QualitySettings,
)
from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke
from workflows.shared.llm_utils.integration_guard import call_text_with_guards

from workflows.enhance.supervision.shared.mini_review import (
    run_mini_review,
)
from workflows.enhance.supervision.shared.prompts import (
    LOOP2_ANALYZER_SYSTEM,
    LOOP2_ANALYZER_USER,
    LOOP2_INTEGRATOR_SYSTEM,
    LOOP2_INTEGRATOR_USER,
    build_word_budget_guidance,
)
from workflows.shared.reference_utils import (
    append_new_references,
    reattach,
    split_references,
)
from workflows.enhance.supervision.shared.types import (
    LiteratureBase,
    LiteratureBaseDecision,
)

logger = logging.getLogger(__name__)

# Allowance applied per Loop 2 iteration — see LOOP1_WORD_ALLOWANCE in
# shared/nodes/integrate_content.py for the overall budget rationale.
LOOP2_WORD_ALLOWANCE_PER_ITER = 2000


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
    explored_bases: Annotated[list[str], add]
    is_complete: bool
    decision: Optional[dict]
    # Error tracking fields
    errors: Annotated[list[dict], add]
    iterations_failed: int
    consecutive_failures: int
    integration_failed: bool
    mini_review_failed: bool
    # Checkpointing (for task queue interruption handling)
    checkpoint_callback: Optional[IncrementalCheckpointCallback]


# =============================================================================
# Nodes
# =============================================================================


async def analyze_for_bases_node(state: Loop2State) -> dict:
    """Analyze review to identify missing literature base."""
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    logger.info(f"Loop 2 iteration {iteration + 1}/{max_iterations}: Analyzing for missing literature bases")

    input_data = state["input"]
    explored_bases_text = "\n".join(f"- {base}" for base in state.get("explored_bases", [])) or "None yet"

    user_prompt = LOOP2_ANALYZER_USER.format(
        review=state["current_review"],
        topic=input_data["topic"],
        research_questions="\n".join(f"- {q}" for q in input_data["research_questions"]),
        explored_bases=explored_bases_text,
        iteration=iteration + 1,
        max_iterations=max_iterations,
    )

    try:
        response = await invoke(
            tier=ModelTier.OPUS,
            system=LOOP2_ANALYZER_SYSTEM,
            user=user_prompt,
            schema=LiteratureBaseDecision,
            config=InvokeConfig(
                max_tokens=2048,
                batch_policy=BatchPolicy.PREFER_SPEED,
            ),
        )

        logger.debug(f"Analyzer decision: {response.action}")
        if response.action == "expand_base":
            logger.info(f"Identified literature base: {response.literature_base.name}")

        return {"decision": response.model_dump()}

    except Exception as e:
        logger.error(f"Loop 2 analysis failed: {e}")
        # Return single-element list - the add reducer will append to existing errors
        return {
            "decision": {
                "action": "error",
                "literature_base": None,
                "reasoning": f"Analysis failed: {e}",
            },
            "errors": [
                {
                    "loop_number": 2,
                    "iteration": iteration,
                    "node_name": "analyze_for_bases",
                    "error_type": "analysis_error",
                    "error_message": str(e),
                    "recoverable": True,
                }
            ],
        }


async def run_mini_review_node(state: Loop2State) -> dict:
    """Execute mini-review on identified literature base."""
    decision = state.get("decision")

    if not decision:
        logger.warning("Mini-review node called without decision")
        # Return single-element list - the add reducer will append to existing errors
        return {
            "mini_review_failed": True,
            "errors": [
                {
                    "loop_number": 2,
                    "iteration": state.get("iteration", 0),
                    "node_name": "run_mini_review",
                    "error_type": "validation_error",
                    "error_message": "No decision provided",
                    "recoverable": False,
                }
            ],
        }

    if decision["action"] != "expand_base":
        logger.warning(f"Mini-review node called with invalid action: {decision['action']}")
        # Return single-element list - the add reducer will append to existing errors
        return {
            "mini_review_failed": True,
            "errors": [
                {
                    "loop_number": 2,
                    "iteration": state.get("iteration", 0),
                    "node_name": "run_mini_review",
                    "error_type": "validation_error",
                    "error_message": f"Expected expand_base action, got: {decision['action']}",
                    "recoverable": False,
                }
            ],
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

    iteration = state.get("iteration", 0)
    iterations_failed = state.get("iterations_failed", 0)

    try:
        decision = state.get("decision")
        if not decision or not decision.get("literature_base"):
            logger.error("Integration node called without valid decision")
            # Return single-element list - the add reducer will append to existing errors
            return {
                "errors": [
                    {
                        "loop_number": 2,
                        "iteration": iteration,
                        "node_name": "integrate_findings",
                        "error_type": "validation_error",
                        "error_message": "No valid decision with literature_base",
                        "recoverable": False,
                    }
                ],
                "iterations_failed": iterations_failed + 1,
                "integration_failed": True,
            }

        literature_base = LiteratureBase(**decision["literature_base"])

        logger.debug(f"Integrating findings for: {literature_base.name}")

        mini_review_text = decision.get("mini_review_text", "")
        new_paper_summaries = decision.get("new_paper_summaries", {})
        new_paper_corpus = decision.get("new_paper_corpus", {})
        new_zotero_keys = decision.get("new_zotero_keys", {})

        # Defensive: ensure merge operands are dicts (guards against upstream
        # returning a list instead of a dict, which causes "'list' object is
        # not a mapping" on the {**a, **b} merge below).
        for name, val in [
            ("paper_summaries", state.get("paper_summaries")),
            ("new_paper_summaries", new_paper_summaries),
            ("zotero_keys", state.get("zotero_keys")),
            ("new_zotero_keys", new_zotero_keys),
            ("paper_corpus", state.get("paper_corpus")),
            ("new_paper_corpus", new_paper_corpus),
        ]:
            if not isinstance(val, dict):
                raise TypeError(
                    f"{name} is {type(val).__name__}, expected dict "
                    f"(value preview: {str(val)[:200]})"
                )

        # Validate mini-review before consuming iteration
        if not mini_review_text or len(mini_review_text.strip()) < 100:
            logger.warning(
                f"Mini-review too short for base '{literature_base.name}' ({len(mini_review_text.strip())} chars)"
            )
            # Return single-element list - the add reducer will append to existing errors
            return {
                # DON'T increment iteration on failure
                "errors": [
                    {
                        "loop_number": 2,
                        "iteration": iteration,
                        "node_name": "integrate_findings",
                        "error_type": "validation_error",
                        "error_message": f"Mini-review failed or too short ({len(mini_review_text.strip()) if mini_review_text else 0} chars)",
                        "recoverable": True,
                    }
                ],
                "iterations_failed": iterations_failed + 1,
                "integration_failed": True,
            }

        citation_keys = "\n".join(
            f"[@{key}] - {new_paper_summaries[doi]['title']}"
            for doi, key in new_zotero_keys.items()
            if doi in new_paper_summaries
        )

        # Operate on prose only: strip trailing references block, let the
        # LLM integrate into the body, then reattach with new entries
        # deterministically appended.
        body, refs_block = split_references(state["current_review"])

        current_words = len(body.split())
        target_words = current_words + LOOP2_WORD_ALLOWANCE_PER_ITER
        word_budget = build_word_budget_guidance(
            current_words=current_words,
            target_words=target_words,
            allowance=LOOP2_WORD_ALLOWANCE_PER_ITER,
            phase_label=f"Loop 2 literature-expansion integration (iteration {iteration + 1})",
        )

        user_prompt = LOOP2_INTEGRATOR_USER.format(
            word_budget=word_budget,
            current_review=body,
            literature_base_name=literature_base.name,
            mini_review=mini_review_text,
            integration_strategy=literature_base.integration_strategy,
            new_citation_keys=citation_keys or "None",
        )

        # Guarded integrator: handles max_tokens continuations and retries
        # if the model self-condenses. Raises IntegrationShrinkageError on
        # unrecoverable shrinkage — we let it propagate (see except below).
        # Note: shrinkage floor is measured on BODY length.
        updated_body = await call_text_with_guards(
            input_content=body,
            tier=ModelTier.OPUS,
            system=LOOP2_INTEGRATOR_SYSTEM,
            user=user_prompt,
            config=InvokeConfig(
                effort="high",
                max_tokens=64000,
                cache=False,
                batch_policy=BatchPolicy.PREFER_SPEED,
            ),
            label=f"loop2_integrator[{literature_base.name[:40]}]",
        )

        merged_summaries = {**state["paper_summaries"], **new_paper_summaries}
        merged_zotero = {**state["zotero_keys"], **new_zotero_keys}

        # Merge paper corpus with full PaperMetadata from mini-review
        merged_corpus = state["paper_corpus"].copy()
        new_papers_added = 0
        for doi, paper_metadata in new_paper_corpus.items():
            if doi not in merged_corpus:
                merged_corpus[doi] = paper_metadata
                new_papers_added += 1

        # Reattach references, appending entries for newly-discovered papers.
        # Uses the merged summaries/keys so newly added papers resolve.
        updated_refs = append_new_references(
            refs_block,
            list(new_paper_corpus.keys()),
            merged_summaries,
            merged_zotero,
        )
        updated_review = reattach(updated_body, updated_refs)

        logger.info(
            f"Integration complete: body {len(body)}→{len(updated_body)} chars "
            f"({current_words}→{len(updated_body.split())} words), "
            f"total_with_refs {len(updated_review)} chars, "
            f"{len(merged_summaries)} total papers, {new_papers_added} new papers"
        )

        # Call checkpoint callback if provided (N=1 for supervision loops)
        # Note: checkpoint needs full accumulated list for resumption
        checkpoint_callback = state.get("checkpoint_callback")
        if checkpoint_callback:
            result = checkpoint_callback(
                iteration + 1,
                {
                    "current_review": updated_review,
                    "iteration": iteration + 1,
                    "explored_bases": state.get("explored_bases", []) + [literature_base.name],
                    "paper_corpus": merged_corpus,
                    "paper_summaries": merged_summaries,
                    "zotero_keys": merged_zotero,
                },
            )
            if hasattr(result, "__await__"):
                await result
            logger.debug(f"Loop 2 checkpoint saved at iteration {iteration + 1}")

        # Return single-element list - the add reducer will append to existing explored_bases
        return {
            "current_review": updated_review,
            "paper_summaries": merged_summaries,
            "zotero_keys": merged_zotero,
            "paper_corpus": merged_corpus,
            "explored_bases": [literature_base.name],
            "iteration": iteration + 1,  # Only increment on success
        }

    except Exception as e:
        # Catch at the lowest level so the last-good current_review (the
        # prior iteration's output) is preserved in state. The loop
        # routing sees integration_failed and finalises; the partial
        # state flows back through run_loop2_node to enhance_report,
        # which then hands it to editing. Errors are surfaced via the
        # errors list and ultimately appear in the task-level
        # "completed with errors" log in workflow_executor.
        logger.error(
            f"Loop 2 integration failed on '{literature_base.name}': {e}. "
            f"Preserving current review (iteration {iteration}).",
            exc_info=True,
        )
        return {
            "integration_failed": True,
            "iterations_failed": iterations_failed + 1,
            "errors": [
                {
                    "loop_number": 2,
                    "iteration": iteration,
                    "node_name": "integrate_findings",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "recoverable": False,
                }
            ],
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

    # Check for failures that didn't increment iteration. On any
    # integration/mini-review failure we finalize immediately — the
    # last-good current_review is already preserved in state, and
    # retrying often costs another expensive Opus call without
    # changing the underlying condition (e.g. JSONDecodeError from
    # truncated CLI stdout that already retried 4x at the CLI layer).
    integration_failed = state.get("integration_failed", False)
    mini_review_failed = state.get("mini_review_failed", False)

    if integration_failed or mini_review_failed:
        logger.warning("Loop 2 failure detected, finalizing with preserved state")
        return "finalize"

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
    checkpoint_callback: IncrementalCheckpointCallback | None = None,
    incremental_state: dict[str, Any] | None = None,
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
        checkpoint_callback: Optional callback for incremental checkpointing.
            Called with (iteration_count, partial_results_dict) after each iteration.
        incremental_state: Optional checkpoint state for resumption.
            Contains iteration_count and partial_results from previous run.

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
    # Handle resume from incremental state
    resumed_iteration = 0
    resumed_review = review
    resumed_explored_bases: list[str] = []
    resumed_paper_corpus = paper_corpus
    resumed_paper_summaries = paper_summaries
    resumed_zotero_keys = zotero_keys

    if incremental_state:
        partial_results = incremental_state.get("partial_results", {})
        resumed_iteration = incremental_state.get("iteration_count", 0)

        if partial_results:
            # Restore state from checkpoint
            resumed_review = partial_results.get("current_review", review)
            resumed_explored_bases = partial_results.get("explored_bases", [])
            resumed_paper_corpus = partial_results.get("paper_corpus", paper_corpus)
            resumed_paper_summaries = partial_results.get("paper_summaries", paper_summaries)
            resumed_zotero_keys = partial_results.get("zotero_keys", zotero_keys)

            logger.info(
                f"Resuming Loop 2 from checkpoint: iteration {resumed_iteration}, "
                f"{len(resumed_explored_bases)} bases already explored"
            )

    initial_state = Loop2State(
        current_review=resumed_review,
        paper_corpus=resumed_paper_corpus,
        paper_summaries=resumed_paper_summaries,
        zotero_keys=resumed_zotero_keys,
        input=input_data,
        quality_settings=quality_settings,
        iteration=resumed_iteration,
        max_iterations=max_iterations,
        explored_bases=resumed_explored_bases,
        is_complete=False,
        decision=None,
        # Error tracking fields
        errors=[],
        iterations_failed=0,
        consecutive_failures=0,
        integration_failed=False,
        mini_review_failed=False,
        # Checkpointing
        checkpoint_callback=checkpoint_callback,
    )

    if resumed_iteration > 0:
        logger.info(
            f"Starting Loop 2 (resumed): iteration={resumed_iteration}/{max_iterations}, "
            f"review length={len(resumed_review)} chars, corpus size={len(resumed_paper_corpus)}"
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
