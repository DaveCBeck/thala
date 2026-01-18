"""Phase 3b: Parallel research workers for web research and book finding."""

import logging
from typing import Any

from langgraph.types import Send

from workflows.research.web_research import deep_research
from workflows.research.book_finding import book_finding
from workflows.wrappers.multi_lang.graph.api import multi_lang_research

logger = logging.getLogger(__name__)


def route_to_parallel_research(state: dict) -> list[Send]:
    """Dispatch parallel web research and book finding workers.

    Creates Send() objects for each generated query and theme,
    allowing them to run in parallel and aggregate results via reducers.
    """
    generated_queries = state.get("generated_queries", [])
    generated_themes = state.get("generated_themes", [])
    input_data = state.get("input", {})
    quality = input_data.get("quality", "standard")
    multi_lang_config = input_data.get("multi_lang_config")

    sends = []

    # Dispatch web research workers
    for i, query_data in enumerate(generated_queries):
        sends.append(
            Send(
                "web_research_worker",
                {
                    "iteration": i,
                    "query": query_data["query"],
                    "quality": quality,
                    "multi_lang_config": multi_lang_config,
                },
            )
        )

    # Dispatch book finding workers
    for i, theme_data in enumerate(generated_themes):
        sends.append(
            Send(
                "book_finding_worker",
                {
                    "iteration": i,
                    "theme": theme_data["theme"],
                    "quality": quality,
                    "multi_lang_config": multi_lang_config,
                },
            )
        )

    logger.info(
        f"Dispatching {len(generated_queries)} web_research + "
        f"{len(generated_themes)} book_finding workers"
    )

    return sends


async def web_research_worker(state: dict) -> dict[str, Any]:
    """Worker that runs a single web research query.

    Receives iteration index and query from Send(), runs deep_research,
    and returns result for aggregation. If multi_lang_config is provided,
    uses multi_lang_research instead.
    """
    iteration = state.get("iteration", 0)
    query = state.get("query", "")
    quality = state.get("quality", "standard")
    multi_lang_config = state.get("multi_lang_config")

    logger.info(f"Web research worker {iteration}: '{query[:50]}...'")

    try:
        if multi_lang_config is not None:
            # Use multi-language research wrapper
            logger.info(f"Multi-lang mode enabled for web research worker {iteration}")
            result_obj = await multi_lang_research(
                topic=query,
                mode=multi_lang_config.get("mode", "set_languages"),
                languages=multi_lang_config.get("languages"),
                workflow="web",
                quality=quality,
            )
            result = result_obj.to_dict()
        else:
            # Direct single-language call
            result = await deep_research(
                query=query,
                quality=quality,
            )

        return {
            "web_research_results": [
                {
                    "iteration": iteration,
                    "query": query,
                    "final_report": result.get("final_report", ""),
                    "source_count": result.get("source_count", 0),
                    "langsmith_run_id": result.get("langsmith_run_id", ""),
                    "status": result.get("status", "unknown"),
                }
            ]
        }

    except Exception as e:
        logger.error(f"Web research worker {iteration} failed: {e}")
        return {
            "web_research_results": [
                {
                    "iteration": iteration,
                    "query": query,
                    "final_report": "",
                    "source_count": 0,
                    "langsmith_run_id": "",
                    "status": "failed",
                }
            ],
            "errors": [
                {"phase": f"web_research_{iteration}", "error": str(e)}
            ],
        }


async def book_finding_worker(state: dict) -> dict[str, Any]:
    """Worker that runs a single book finding theme.

    Receives iteration index and theme from Send(), runs book_finding,
    and returns result for aggregation. If multi_lang_config is provided,
    uses multi_lang_research instead.
    """
    iteration = state.get("iteration", 0)
    theme = state.get("theme", "")
    quality = state.get("quality", "standard")
    multi_lang_config = state.get("multi_lang_config")

    logger.info(f"Book finding worker {iteration}: '{theme[:50]}...'")

    try:
        if multi_lang_config is not None:
            # Use multi-language research wrapper
            logger.info(f"Multi-lang mode enabled for book finding worker {iteration}")
            result_obj = await multi_lang_research(
                topic=theme,
                mode=multi_lang_config.get("mode", "set_languages"),
                languages=multi_lang_config.get("languages"),
                workflow="books",
                quality=quality,
            )
            result = result_obj.to_dict()
        else:
            # Direct single-language call
            result = await book_finding(
                theme=theme,
                quality=quality,
            )

        # Extract zotero keys from processed books
        # Note: After our Part 1 changes, processed_books now have zotero_key
        # We need to load the full result from the workflow state store
        # For now, we'll extract what we can from the result
        zotero_keys = []

        return {
            "book_finding_results": [
                {
                    "iteration": iteration,
                    "theme": theme,
                    "final_report": result.get("final_report", ""),
                    "processed_books": [],  # Would need to load from state store
                    "zotero_keys": zotero_keys,
                    "status": result.get("status", "unknown"),
                }
            ]
        }

    except Exception as e:
        logger.error(f"Book finding worker {iteration} failed: {e}")
        return {
            "book_finding_results": [
                {
                    "iteration": iteration,
                    "theme": theme,
                    "final_report": "",
                    "processed_books": [],
                    "zotero_keys": [],
                    "status": "failed",
                }
            ],
            "errors": [
                {"phase": f"book_finding_{iteration}", "error": str(e)}
            ],
        }


async def aggregate_research(state: dict) -> dict[str, Any]:
    """Aggregate results from parallel research workers.

    This node runs after all workers complete (due to reducer aggregation).
    It logs summary statistics and transitions to the synthesis phase.
    """
    web_results = state.get("web_research_results", [])
    book_results = state.get("book_finding_results", [])

    successful_web = sum(1 for r in web_results if r.get("status") == "success")
    successful_books = sum(1 for r in book_results if r.get("status") == "success")

    total_web_sources = sum(r.get("source_count", 0) for r in web_results)

    logger.info(
        f"Research aggregation: {successful_web}/{len(web_results)} web research, "
        f"{successful_books}/{len(book_results)} book finding. "
        f"Total web sources: {total_web_sources}"
    )

    return {
        "current_phase": "synthesis",
    }
