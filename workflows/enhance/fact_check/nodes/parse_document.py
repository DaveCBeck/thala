"""Parse document node for fact-check workflow."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.nodes.parse_document import parse_markdown_to_model

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="FactCheckParseDocument")
async def parse_document_node(state: dict) -> dict[str, Any]:
    """Parse document or use provided document model.

    If a document_model is already provided in input, skip parsing.
    Otherwise, parse the document markdown into a structured model.

    Args:
        state: Current workflow state

    Returns:
        State update with document_model and parse_complete
    """
    input_data = state.get("input", {})
    document_model_dict = input_data.get("document_model")

    # If document model already provided, use it directly
    if document_model_dict:
        logger.debug("Using pre-provided document model, skipping parse")
        return {
            "document_model": document_model_dict,
            "parse_complete": True,
            "parse_warnings": [],
        }

    # Otherwise, parse the document
    document = input_data.get("document", "")
    if not document:
        logger.error("No document or document_model provided")
        return {
            "document_model": {},
            "parse_complete": True,
            "parse_warnings": ["No document provided"],
            "errors": [{"node": "parse_document", "error": "No document provided"}],
        }

    logger.debug(f"Parsing document ({len(document)} chars)")

    try:
        # Use the same parser as the editing workflow
        document_model = parse_markdown_to_model(document)

        logger.info(
            f"Parsed document: {document_model.section_count} sections, "
            f"{document_model.block_count} blocks, {document_model.total_words} words"
        )

        return {
            "document_model": document_model.to_dict(),
            "parse_complete": True,
            "parse_warnings": [],
        }

    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        return {
            "document_model": {},
            "parse_complete": True,
            "parse_warnings": [f"Parse error: {e}"],
            "errors": [{"node": "parse_document", "error": str(e)}],
        }
