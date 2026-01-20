"""Parse document node for editing workflow."""

import logging
from typing import Any

from langsmith import traceable

from workflows.enhance.editing.parser import parse_markdown_to_model, validate_document_model

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="EditingParseDocument")
async def parse_document_node(state: dict) -> dict[str, Any]:
    """Parse markdown document into structured DocumentModel.

    Args:
        state: Current workflow state with input.document

    Returns:
        State update with document_model and parse metadata
    """
    document_text = state["input"]["document"]

    logger.info(f"Parsing document: {len(document_text)} chars")

    # Parse into structured model
    document_model = parse_markdown_to_model(document_text)

    # Validate model
    warnings = validate_document_model(document_model)
    if warnings:
        for warning in warnings:
            logger.warning(f"Parse warning: {warning}")

    logger.info(
        f"Parsed: {document_model.total_words} words, "
        f"{document_model.section_count} sections, "
        f"{document_model.block_count} blocks"
    )

    return {
        "document_model": document_model.to_dict(),
        "parse_complete": True,
        "parse_warnings": warnings,
    }
