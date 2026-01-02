"""
Save to top_of_mind node.

Saves 4 records to top_of_mind vector store:
1. Web research output
2. Academic literature review output
3. Book recommendations output
4. Combined summary

Each record includes metadata for filtering and search.
"""

import logging
from typing import Any
from uuid import uuid4

from workflows.wrapped.state import WrappedResearchState

logger = logging.getLogger(__name__)


async def save_to_top_of_mind(state: WrappedResearchState) -> dict[str, Any]:
    """Save all outputs to top_of_mind vector store.

    Creates 4 separate records:
    - web: Web research findings
    - academic: Academic literature review
    - books: Book recommendations
    - combined: Synthesized summary of all three

    Each record includes:
    - content: The markdown output
    - metadata: source type, original query, workflow info
    - embedding: Auto-generated for semantic search
    """
    try:
        from core.stores.schema import BaseRecord, SourceType
        from langchain_tools.base import get_store_manager

        store_manager = get_store_manager()
    except ImportError as e:
        logger.warning(f"Store components not available: {e}")
        return {
            "top_of_mind_ids": {},
            "current_phase": "saved_to_top_of_mind",
            "errors": [{"phase": "save_to_top_of_mind", "error": f"Store not available: {e}"}],
        }

    top_of_mind_ids: dict[str, str] = {}
    errors: list[dict] = []

    query = state["input"]["query"]
    langsmith_run_id = state.get("langsmith_run_id")

    # Prepare records to save
    records_to_save = [
        ("web", state.get("web_result", {}).get("final_output")),
        ("academic", state.get("academic_result", {}).get("final_output")),
        ("books", state.get("book_result", {}).get("final_output")),
        ("combined", state.get("combined_summary")),
    ]

    for workflow_type, content in records_to_save:
        if not content:
            logger.warning(f"No content for {workflow_type}, skipping")
            continue

        try:
            # Generate embedding (take first 8000 chars to avoid token limits)
            embed_text = content[:8000]
            embedding = await store_manager.embedding.embed(embed_text)

            record = BaseRecord(
                id=uuid4(),
                source_type=SourceType.INTERNAL,
                content=content,
                metadata={
                    "type": "wrapped_research",
                    "workflow": workflow_type,
                    "original_query": query,
                    "langsmith_run_id": langsmith_run_id,
                    "quality": state["input"]["quality"],
                },
                embedding_model=store_manager.embedding.model,
            )

            # Save to ChromaDB (top_of_mind)
            await store_manager.chroma.add(
                record=record,
                embedding=embedding,
                document=content,
            )

            top_of_mind_ids[workflow_type] = str(record.id)
            logger.info(f"Saved {workflow_type} to top_of_mind: {record.id}")

        except Exception as e:
            logger.error(f"Failed to save {workflow_type}: {e}")
            errors.append({"phase": f"save_{workflow_type}", "error": str(e)})

    result: dict[str, Any] = {
        "top_of_mind_ids": top_of_mind_ids,
        "current_phase": "saved_to_top_of_mind",
    }

    if errors:
        result["errors"] = errors

    logger.info(f"Saved {len(top_of_mind_ids)} records to top_of_mind")
    return result
