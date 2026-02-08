"""SVG validation and sanitization utilities.

Provides functions to validate SVG XML well-formedness and sanitize
unescaped XML entities in text content that would cause parsing failures.
"""

import logging
import re

from lxml import etree

logger = logging.getLogger(__name__)


def strip_code_fences(content: str) -> str:
    """Remove markdown code fences wrapping SVG content."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```svg or ```) and last line (```)
        if lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1])
        else:
            content = "\n".join(lines[1:])
    return content.strip()


def validate_and_sanitize_svg(svg_string: str, width: int = 800, height: int = 600) -> str | None:
    """Validate SVG and attempt repairs before rendering.

    Unified pipeline that replaces separate validate_svg_xml() +
    sanitize_svg_text_entities() calls. Uses lxml's recover mode to
    handle malformed XML gracefully.

    Args:
        svg_string: Raw SVG string (may include code fences, broken XML)
        width: Default width for viewBox if missing
        height: Default height for viewBox if missing

    Returns:
        Sanitized SVG string, or None if completely broken.
    """
    svg_string = strip_code_fences(svg_string)

    # Pre-sanitize XML entities before parsing
    svg_string = sanitize_svg_text_entities(svg_string)

    # Extract <svg>...</svg> if there's surrounding content
    if not svg_string.strip().startswith("<svg"):
        svg_start = svg_string.find("<svg")
        svg_end = svg_string.rfind("</svg>")
        if svg_start == -1 or svg_end == -1:
            return None
        svg_string = svg_string[svg_start : svg_end + 6]

    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=True)
        tree = etree.fromstring(svg_string.encode(), parser)
    except etree.XMLSyntaxError:
        return None

    if tree is None:
        return None

    # Ensure required attributes
    if tree.get("xmlns") is None:
        tree.set("xmlns", "http://www.w3.org/2000/svg")
    if tree.get("viewBox") is None:
        w = tree.get("width", str(width))
        h = tree.get("height", str(height))
        tree.set("viewBox", f"0 0 {w} {h}")

    return etree.tostring(tree, encoding="unicode", pretty_print=True)


def validate_svg_xml(svg_content: str) -> tuple[bool, str | None]:
    """Check if SVG content is well-formed XML.

    Args:
        svg_content: Raw SVG string

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
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
    svg_content = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", svg_content)

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
    "validate_and_sanitize_svg",
    "validate_svg_xml",
    "sanitize_svg_text_entities",
    "strip_code_fences",
    "extract_validation_error_type",
]
