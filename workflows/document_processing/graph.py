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
import uuid
from datetime import datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from workflows.document_processing.nodes import (
    check_metadata,
    create_zotero_stub,
    detect_document_language,
    generate_summary,
    resolve_input,
    save_short_summary,
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

logger = logging.getLogger(__name__)


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
    builder.add_node("update_store", update_store)
    builder.add_node("detect_language", detect_document_language)
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("check_metadata", check_metadata)
    builder.add_node("save_short_summary", save_short_summary)
    builder.add_node("update_zotero", update_zotero)
    builder.add_node("detect_chapters", detect_chapters)
    builder.add_node("chapter_summarization", chapter_summarization_subgraph)
    builder.add_node("save_tenth_summary", save_tenth_summary)
    builder.add_node("finalize", finalize)

    # Linear flow: resolve_input now produces processing_result directly
    builder.add_edge(START, "resolve_input")
    builder.add_edge("resolve_input", "create_zotero_stub")
    builder.add_edge("create_zotero_stub", "update_store")

    # Language detection before summary generation
    builder.add_edge("update_store", "detect_language")

    builder.add_conditional_edges(
        "detect_language",
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
    langs: list[str] = None,
    extra_metadata: dict = None,
    use_batch_api: bool = True,  # Set False for rapid iteration (skips batch API)
) -> dict[str, Any]:
    """Process a document through the full pipeline."""
    graph = create_document_processing_graph()

    initial_state = {
        "input": {
            "source": source,
            "title": title,
            "item_type": item_type,
            "langs": langs or ["English"],
            "extra_metadata": extra_metadata or {},
            "use_batch_api": use_batch_api,
        },
        "store_records": [],
        "errors": [],
        "chapters": [],
        "chapter_summaries": [],
        "metadata_updates": {},
        "current_status": "starting",
        "started_at": datetime.utcnow(),
    }

    run_id = uuid.uuid4()
    desc = title or (source[:30] if isinstance(source, str) else "document")
    logger.info(f"Starting document processing for: {title or source[:100]}")
    logger.info(f"LangSmith run ID: {run_id}")
    result = await graph.ainvoke(
        initial_state,
        config={
            "run_id": run_id,
            "run_name": f"doc:{desc[:30]}",
        },
    )
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
                langs=doc_config.get("langs"),
                extra_metadata=doc_config.get("extra_metadata"),
            )

    logger.info(
        f"Starting batch processing of {len(documents)} documents (concurrency: {concurrency})"
    )

    tasks = [process_with_limit(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, (doc, result) in enumerate(zip(documents, results)):
        if isinstance(result, Exception):
            logger.warning(f"Document failed: {result}")
            processed_results.append(
                {
                    "input": doc,
                    "current_status": "failed",
                    "errors": [{"node": "batch_processor", "error": str(result)}],
                }
            )
        else:
            processed_results.append(result)

    succeeded = sum(1 for r in processed_results if r.get("current_status") != "failed")
    logger.info(f"Batch processing complete: {succeeded}/{len(documents)} succeeded")

    return processed_results
