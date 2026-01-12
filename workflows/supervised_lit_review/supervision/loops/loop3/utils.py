"""Result formatting and helper functions for loop3."""

import logging
from typing import Any

from workflows.supervised_lit_review.supervision.utils import (
    number_paragraphs,
    strip_paragraph_numbers,
)

logger = logging.getLogger(__name__)


def number_paragraphs_node(state: dict) -> dict[str, Any]:
    """Add paragraph numbers to document for structural editing."""
    current_review = state["current_review"]

    if not current_review:
        logger.warning("No review content to number")
        return {
            "is_complete": True,
            "numbered_document": "",
            "paragraph_mapping": {},
        }

    numbered_doc, para_mapping = number_paragraphs(current_review)

    logger.info(f"Numbered document into {len(para_mapping)} paragraphs")

    return {
        "numbered_document": numbered_doc,
        "paragraph_mapping": para_mapping,
    }


def validate_result_node(state: dict) -> dict[str, Any]:
    """Strip paragraph numbers and validate the restructured output."""
    current_review = state["current_review"]

    cleaned_review = strip_paragraph_numbers(current_review)

    if not cleaned_review or len(cleaned_review) < 100:
        logger.warning("Validation failed: output too short or empty")
        original_numbered = state.get("numbered_document", "")
        if original_numbered:
            cleaned_review = strip_paragraph_numbers(original_numbered)

    logger.info(f"Validated result: {len(cleaned_review)} characters")

    return {
        "current_review": cleaned_review,
    }


def increment_iteration(state: dict) -> dict[str, Any]:
    """Increment iteration counter for next loop."""
    return {
        "iteration": state["iteration"] + 1,
    }


def finalize_node(state: dict) -> dict[str, Any]:
    """Mark loop as complete with accurate iteration count.

    Correctly counts iterations when work was done, fixing the
    "0 iterations used" bug when we actually did analysis and editing.
    """
    iteration = state.get("iteration", 0)
    edit_manifest = state.get("edit_manifest")
    phase_a_complete = state.get("phase_a_complete", False)

    did_work = (
        phase_a_complete or
        (edit_manifest is not None and (
            len(edit_manifest.get("edits", [])) > 0 or
            len(edit_manifest.get("todo_markers", [])) > 0
        ))
    )

    actual_iterations = max(iteration, 1) if did_work else iteration

    logger.info(f"Loop 3 complete: {actual_iterations} iterations used")
    return {
        "is_complete": True,
        "iteration": actual_iterations,
    }
