"""Edit validation and programmatic application."""

import logging
from typing import Any

from workflows.supervised_lit_review.supervision.types import StructuralEdit
from workflows.supervised_lit_review.supervision.utils import (
    validate_structural_edits,
    apply_structural_edits,
    verify_edits_applied,
)

logger = logging.getLogger(__name__)


def validate_edits_node(state: dict) -> dict[str, Any]:
    """Validate that structural edits reference valid paragraphs."""
    manifest = state.get("edit_manifest")
    paragraph_mapping = state.get("paragraph_mapping", {})

    if not manifest or not manifest.get("edits"):
        logger.debug("No edits to validate")
        return {
            "valid_edits": [],
            "invalid_edits": [],
            "needs_retry_edits": [],
            "validation_errors": {},
        }

    edits = [StructuralEdit(**e) for e in manifest.get("edits", [])]

    result = validate_structural_edits(paragraph_mapping, edits)

    logger.info(
        f"Edit validation complete: {len(result['valid_edits'])} valid, "
        f"{len(result['invalid_edits'])} invalid, "
        f"{len(result['needs_retry_edits'])} need retry"
    )

    if result["invalid_edits"]:
        for idx, error in result["errors"].items():
            logger.warning(f"Invalid edit {idx}: {error}")

    if result["needs_retry_edits"]:
        for edit in result["needs_retry_edits"]:
            logger.debug(f"Edit needs retry (missing replacement_text): P{edit.source_paragraph}")

    return {
        "valid_edits": [e.model_dump() for e in result["valid_edits"]],
        "invalid_edits": [e.model_dump() for e in result["invalid_edits"]],
        "needs_retry_edits": [e.model_dump() for e in result["needs_retry_edits"]],
        "validation_errors": result["errors"],
    }


def apply_edits_programmatically_node(state: dict) -> dict[str, Any]:
    """Apply validated edits programmatically using paragraph mapping."""
    paragraph_mapping = state.get("paragraph_mapping", {})
    valid_edits = state.get("valid_edits", [])

    if not valid_edits:
        logger.debug("No valid edits to apply programmatically")
        return {"fallback_used": False}

    edits = [StructuralEdit(**e) for e in valid_edits]

    restructured, applied_descriptions = apply_structural_edits(
        paragraph_mapping, edits
    )

    logger.info(f"Programmatically applied {len(applied_descriptions)} edits")
    for desc in applied_descriptions:
        logger.debug(f"  - {desc}")

    return {
        "current_review": restructured,
        "applied_edits": applied_descriptions,
        "fallback_used": False,
    }


def verify_application_node(state: dict) -> dict[str, Any]:
    """Verify that edits were actually applied to the document."""
    original_mapping = state.get("paragraph_mapping", {})
    current_review = state.get("current_review", "")
    valid_edits = state.get("valid_edits", [])

    if not valid_edits:
        return {}

    edits = [StructuralEdit(**e) for e in valid_edits]
    verifications = verify_edits_applied(original_mapping, current_review, edits)

    failed = [k for k, v in verifications.items() if not v]
    if failed:
        logger.warning(f"Edit verification failures: {failed}")
    else:
        logger.debug(f"All {len(verifications)} edits verified as applied")

    return {}
