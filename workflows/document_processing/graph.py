"""
Main document processing workflow graph.

Uses Anthropic Claude models for LLM tasks:
- Sonnet: Standard summarization and metadata extraction
- Opus with extended thinking: Complex chapter analysis

Supports batch processing for 50% cost reduction when processing
multiple documents together.
"""

from core.config import configure_langsmith

configure_langsmith()

import asyncio
import logging
from datetime import datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from workflows.document_processing.nodes import (
    check_metadata,
    create_zotero_stub,
    generate_summary,
    process_marker,
    resolve_input,
    save_short_summary,
    smart_chunker,
    update_store,
    update_zotero,
)
from workflows.document_processing.nodes.chapter_detector import detect_chapters
from workflows.document_processing.nodes.finalizer import finalize
from workflows.document_processing.nodes.save_tenth_summary import save_tenth_summary
from workflows.document_processing.state import DocumentProcessingState
from workflows.document_processing.subgraphs.chapter_summarization import (
    chapter_summarization_subgraph,
)
from workflows.shared.async_utils import gather_with_error_collection

logger = logging.getLogger(__name__)


def route_by_source_type(state: DocumentProcessingState) -> str:
    """Route based on whether input is already markdown."""
    if state.get("is_already_markdown"):
        return "markdown"
    return "needs_marker"


def fan_out_to_agents(state: DocumentProcessingState) -> list[Send]:
    """Fan out to parallel summary and metadata agents using Send()."""
    return [
        Send("generate_summary", state),
        Send("check_metadata", state),
    ]


def route_by_doc_size(state: DocumentProcessingState) -> str:
    """Route based on whether doc needs 10:1 summary."""
    needs_tenth = state.get("needs_tenth_summary", False)
    if needs_tenth:
        return "needs_tenth"
    return "skip_tenth"


def create_document_processing_graph():
    """Create the main document processing graph."""
    builder = StateGraph(DocumentProcessingState)

    builder.add_node("resolve_input", resolve_input)
    builder.add_node("create_zotero_stub", create_zotero_stub)
    builder.add_node(
        "process_marker",
        process_marker,
        retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
    )
    builder.add_node("smart_chunker", smart_chunker)
    builder.add_node("update_store", update_store)
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("check_metadata", check_metadata)
    builder.add_node("save_short_summary", save_short_summary)
    builder.add_node("update_zotero", update_zotero)
    builder.add_node("detect_chapters", detect_chapters)
    builder.add_node("chapter_summarization", chapter_summarization_subgraph)
    builder.add_node("save_tenth_summary", save_tenth_summary)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "resolve_input")
    builder.add_edge("resolve_input", "create_zotero_stub")

    builder.add_conditional_edges(
        "create_zotero_stub",
        route_by_source_type,
        {
            "markdown": "smart_chunker",
            "needs_marker": "process_marker",
        },
    )

    builder.add_edge("smart_chunker", "update_store")
    builder.add_edge("process_marker", "update_store")

    builder.add_conditional_edges(
        "update_store",
        fan_out_to_agents,
        ["generate_summary", "check_metadata"],
    )

    builder.add_edge("generate_summary", "save_short_summary")
    builder.add_edge("check_metadata", "save_short_summary")

    builder.add_edge("save_short_summary", "update_zotero")

    builder.add_edge("update_zotero", "detect_chapters")
    builder.add_conditional_edges(
        "detect_chapters",
        route_by_doc_size,
        {
            "needs_tenth": "chapter_summarization",
            "skip_tenth": "finalize",
        },
    )

    builder.add_edge("chapter_summarization", "save_tenth_summary")
    builder.add_edge("save_tenth_summary", "finalize")

    builder.add_edge("finalize", END)

    return builder.compile()


async def process_document(
    source: str,
    title: str = None,
    item_type: str = "document",
    quality: str = "balanced",
    langs: list[str] = None,
    extra_metadata: dict = None,
) -> dict[str, Any]:
    """Process a document through the full pipeline."""
    graph = create_document_processing_graph()

    initial_state = {
        "input": {
            "source": source,
            "title": title,
            "item_type": item_type,
            "quality": quality,
            "langs": langs or ["English"],
            "extra_metadata": extra_metadata or {},
        },
        "store_records": [],
        "errors": [],
        "chapters": [],
        "chapter_summaries": [],
        "metadata_updates": {},
        "current_status": "starting",
        "started_at": datetime.utcnow(),
    }

    logger.info(f"Starting document processing for: {source}")
    result = await graph.ainvoke(initial_state)
    logger.info(f"Document processing complete. Status: {result.get('current_status')}")

    return result


async def process_documents_batch(
    documents: list[dict[str, Any]],
    concurrency: int = 5,
) -> list[dict[str, Any]]:
    """Process multiple documents with concurrent execution."""
    semaphore = asyncio.Semaphore(concurrency)

    async def process_with_limit(doc_config: dict) -> dict[str, Any]:
        async with semaphore:
            return await process_document(
                source=doc_config["source"],
                title=doc_config.get("title"),
                item_type=doc_config.get("item_type", "document"),
                quality=doc_config.get("quality", "balanced"),
                langs=doc_config.get("langs"),
                extra_metadata=doc_config.get("extra_metadata"),
            )

    logger.info(f"Starting batch processing of {len(documents)} documents (concurrency: {concurrency})")

    tasks = [process_with_limit(doc) for doc in documents]
    successes, errors = await gather_with_error_collection(tasks, logger, error_template="Document failed: {error}")

    processed_results = []
    for i, doc in enumerate(documents):
        error_match = next((e for e in errors if e["index"] == i), None)
        if error_match:
            processed_results.append({
                "input": doc,
                "current_status": "failed",
                "errors": [{"node": "batch_processor", "error": error_match["error"]}],
            })
        else:
            result_idx = i - sum(1 for e in errors if e["index"] < i)
            processed_results.append(successes[result_idx])

    succeeded = sum(1 for r in processed_results if r.get("current_status") != "failed")
    logger.info(f"Batch processing complete: {succeeded}/{len(documents)} succeeded")

    return processed_results
