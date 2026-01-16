"""Content-based language filtering for research results.

Provides filtering based on text content (abstract, title) rather than
unreliable metadata. Use as a secondary check after query-based filtering
to catch papers where metadata indicates target language but content differs.

Example usage:
    from workflows.shared.language import filter_by_content_language

    # Filter papers by abstract language
    matching, rejected = filter_by_content_language(
        papers,
        target_language="es",
        text_fields=["abstract", "title"],
    )
"""

import logging
from typing import Any

from .detection import detect_language, DEFAULT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


def filter_by_content_language(
    items: list[dict[str, Any]],
    target_language: str,
    text_fields: list[str] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter items by detected language from text content.

    Args:
        items: List of dicts with text fields (papers, sources, etc.)
        target_language: ISO 639-1 code (e.g., "es", "de", "zh")
        text_fields: Fields to check for language detection, in priority order.
                     Defaults to ["abstract", "title"].
        confidence_threshold: Minimum confidence to accept detection (default 0.7)

    Returns:
        Tuple of (matching, rejected):
        - matching: Items that match target language or couldn't be determined
        - rejected: Items detected as different language with high confidence

    Note:
        - Items with no detectable text are kept (benefit of doubt)
        - Skip calling this for English target - no filtering needed
        - Uses first field with sufficient text for detection
    """
    if text_fields is None:
        text_fields = ["abstract", "title"]

    if not items:
        return [], []

    matching: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in items:
        # Try to get text from fields in priority order
        text_sample = ""
        for field in text_fields:
            field_value = item.get(field, "")
            if field_value and len(field_value) > 20:  # Need meaningful text
                text_sample = field_value[:1000]  # Limit for detection
                break

        # No text available - keep item (benefit of doubt)
        if not text_sample:
            matching.append(item)
            continue

        # Detect language
        detected_lang, confidence = detect_language(text_sample)

        # Detection failed - keep item
        if detected_lang is None:
            matching.append(item)
            continue

        # Check if matches target
        if detected_lang == target_language and confidence >= confidence_threshold:
            matching.append(item)
        elif detected_lang != target_language and confidence >= confidence_threshold:
            # High confidence it's a different language - reject
            rejected.append(item)
            logger.debug(
                f"Rejected item: detected {detected_lang} ({confidence:.2f}), "
                f"expected {target_language}"
            )
        else:
            # Low confidence - keep item
            matching.append(item)

    return matching, rejected
