"""Input validation node for substack_review workflow."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

CITATION_PATTERN = r"\[@([^\]]+)\]"
MIN_WORD_COUNT = 500


async def validate_input_node(state: dict) -> dict[str, Any]:
    """Validate input and extract citation keys.

    Checks:
    - Literature review has minimum word count
    - Extracts all [@KEY] citations for later reference lookup

    Returns:
        State update with is_valid, validation_error, extracted_citation_keys
    """
    lit_review = state["input"]["literature_review"]

    # Check minimum length
    word_count = len(lit_review.split())
    if word_count < MIN_WORD_COUNT:
        logger.warning(f"Literature review too short: {word_count} words")
        return {
            "is_valid": False,
            "validation_error": (
                f"Literature review too short ({word_count} words). "
                f"Minimum {MIN_WORD_COUNT} words required."
            ),
            "extracted_citation_keys": [],
        }

    # Extract citation keys
    matches = re.findall(CITATION_PATTERN, lit_review)
    keys = set()
    for match in matches:
        # Handle multi-citations like [@key1; @key2]
        for key in match.split(";"):
            key = key.strip().lstrip("@")
            if key:
                keys.add(key)

    citation_keys = sorted(keys)

    if not citation_keys:
        logger.warning("No citations found in literature review")

    logger.info(
        f"Validated input: {word_count} words, {len(citation_keys)} unique citations"
    )

    return {
        "is_valid": True,
        "validation_error": None,
        "extracted_citation_keys": citation_keys,
    }
