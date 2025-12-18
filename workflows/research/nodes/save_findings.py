"""
Save findings node.

Saves the final research report to Thala's store.
Creates a StoreRecord with embedding for semantic search.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from workflows.research.state import DeepResearchState

logger = logging.getLogger(__name__)


async def save_findings(state: DeepResearchState) -> dict[str, Any]:
    """Save final research report to the store.

    Creates a StoreRecord with:
    - compression_level=0 (original research)
    - metadata including topic, sources, citations
    - embedding for semantic search

    Returns:
        - store_record_id: UUID of saved record
        - completed_at: Completion timestamp
        - current_status: "completed"
    """
    final_report = state.get("final_report")
    if not final_report:
        logger.warning("No final report to save")
        return {
            "current_status": "completed",
            "completed_at": datetime.utcnow(),
            "errors": [{"node": "save_findings", "error": "No final report"}],
        }

    brief = state.get("research_brief", {})
    citations = state.get("citations", [])
    diffusion = state.get("diffusion", {})

    try:
        # Import store components
        from core.stores.schema import SourceType, StoreRecord
        from langchain_tools.base import get_store_manager

        record_id = uuid4()

        record = StoreRecord(
            id=record_id,
            source_type=SourceType.INTERNAL,
            content=final_report,
            compression_level=0,  # Original research
            metadata={
                "type": "research_report",
                "topic": brief.get("topic", "Unknown"),
                "objectives": brief.get("objectives", []),
                "source_count": len(citations),
                "depth": state.get("input", {}).get("depth", "standard"),
                "iterations": diffusion.get("iteration", 0),
                "completeness": diffusion.get("completeness_score", 0),
                "citations": [c.get("url") for c in citations],
            },
        )

        store_manager = get_store_manager()

        # Generate embedding for semantic search
        # Take first 8000 chars to avoid token limits
        embed_text = final_report[:8000]
        embedding = await store_manager.embedding.embed(embed_text)
        record.embedding = embedding
        record.embedding_model = store_manager.embedding.model

        # Save to Elasticsearch store
        await store_manager.es_stores.store.add(record)

        logger.info(f"Saved research report to store: {record_id}")

        return {
            "store_record_id": str(record_id),
            "current_status": "completed",
            "completed_at": datetime.utcnow(),
        }

    except ImportError as e:
        logger.warning(f"Store components not available: {e}")
        # Continue without saving - the report is still in state
        return {
            "current_status": "completed",
            "completed_at": datetime.utcnow(),
            "errors": [{"node": "save_findings", "error": f"Store not available: {e}"}],
        }

    except Exception as e:
        logger.error(f"Failed to save research: {e}")
        return {
            "current_status": "completed",
            "completed_at": datetime.utcnow(),
            "errors": [{"node": "save_findings", "error": str(e)}],
        }
