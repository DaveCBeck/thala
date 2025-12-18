"""
Summary generation node for document processing workflow.
"""

import logging
from typing import Any

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.llm_utils import ModelTier, summarize_text
from workflows.shared.text_utils import get_first_n_pages, get_last_n_pages

logger = logging.getLogger(__name__)


async def generate_summary(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Generate 100-word summary of the document.

    For very long docs (>50k chars): uses first+last 10 pages.
    Returns short_summary and updated current_status.
    """
    try:
        processing_result = state.get("processing_result")
        if not processing_result:
            logger.error("No processing_result in state")
            return {
                "errors": [{"node": "summary_agent", "error": "No processing result"}],
            }

        markdown = processing_result["markdown"]

        # For very long documents, use first and last pages
        if len(markdown) > 50000:
            logger.info("Document is long, using first and last 10 pages for summary")
            first_pages = get_first_n_pages(markdown, 10)
            last_pages = get_last_n_pages(markdown, 10)
            content = f"{first_pages}\n\n[... middle section omitted ...]\n\n{last_pages}"
        else:
            content = markdown

        # Generate summary via LLM (Sonnet for standard summarization)
        context = "Create a concise summary capturing the main thesis, key arguments, and conclusions. Focus on what makes this work significant and its core contributions."
        summary = await summarize_text(
            text=content,
            target_words=100,
            context=context,
            tier=ModelTier.SONNET,
        )

        logger.info(f"Generated summary ({len(summary.split())} words)")

        # Don't update current_status here - parallel nodes would conflict
        return {
            "short_summary": summary.strip(),
        }

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return {
            "errors": [{"node": "summary_agent", "error": str(e)}],
        }
