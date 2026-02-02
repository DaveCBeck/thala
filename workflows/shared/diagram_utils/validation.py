"""SVG validation and sanitization utilities.

Provides functions to validate SVG XML well-formedness and sanitize
unescaped XML entities in text content that would cause parsing failures.
"""

import logging
import re

logger = logging.getLogger(__name__)


def validate_svg_xml(svg_content: str) -> tuple[bool, str | None]:
    """Check if SVG content is well-formed XML.

    Args:
        svg_content: Raw SVG string

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml not installed")
        return True, None  # Can't validate without lxml, assume valid

    try:
        etree.fromstring(svg_content.encode())
        return True, None
    except etree.XMLSyntaxError as e:
        return False, str(e)


def sanitize_svg_text_entities(svg_content: str) -> str:
    """Sanitize unescaped XML entities in SVG text content.

    Fixes common issues where LLM-generated SVG contains unescaped
    special characters like & or < in text elements.

    Args:
        svg_content: Raw SVG string (potentially invalid XML)

    Returns:
        Sanitized SVG string with escaped entities in text content
    """
    # Step 1: Escape & that aren't already part of entity references
    # This is safe to do globally since valid entities are preserved
    svg_content = re.sub(
        r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", svg_content
    )

    # Step 2: Escape < that are clearly not tag starts
    # A < is a tag start if followed by: letter, /, !, or ?
    # Anything else (space, digit, etc.) is likely a comparison operator in text
    svg_content = re.sub(r"<(?![a-zA-Z/?!])", "&lt;", svg_content)

    return svg_content


def extract_validation_error_type(error: Exception) -> str:
    """Categorize XML validation errors for better logging.

    Args:
        error: Exception from XML parsing

    Returns:
        Error category string
    """
    error_str = str(error).lower()

    if "xmlparseentityref" in error_str or "no name" in error_str:
        return "unescaped_entity"
    if "not well-formed" in error_str or "invalid token" in error_str:
        return "malformed_xml"
    if "namespace" in error_str:
        return "namespace_error"
    if "encoding" in error_str:
        return "encoding_error"

    return "unknown_xml_error"


__all__ = [
    "validate_svg_xml",
    "sanitize_svg_text_entities",
    "extract_validation_error_type",
]
