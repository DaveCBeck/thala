"""SVG to PNG conversion utilities.

Provides functions for converting SVG content to PNG using CairoSVG.
"""

import logging

from .validation import (
    extract_validation_error_type,
    sanitize_svg_text_entities,
    validate_svg_xml,
)

logger = logging.getLogger(__name__)


def convert_svg_to_png(
    svg_content: str,
    dpi: int = 150,
    background_color: str = "#ffffff",
) -> bytes | None:
    """Convert SVG to PNG using CairoSVG.

    Args:
        svg_content: Raw SVG string
        dpi: Output DPI (default 150)
        background_color: Background color for transparency handling

    Returns:
        PNG bytes if successful, None on failure
    """
    try:
        import cairosvg
    except ImportError:
        logger.error("cairosvg not installed. Run: pip install cairosvg")
        return None

    # Validate and sanitize SVG before conversion
    is_valid, error = validate_svg_xml(svg_content)
    if not is_valid:
        error_type = extract_validation_error_type(Exception(error or ""))
        logger.warning(
            f"SVG validation failed ({error_type}): {error}, attempting sanitization"
        )
        svg_content = sanitize_svg_text_entities(svg_content)

        # Re-validate after sanitization
        is_valid_after, error_after = validate_svg_xml(svg_content)
        if not is_valid_after:
            logger.error(f"SVG still invalid after sanitization: {error_after}")

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            dpi=dpi,
            background_color=background_color,
        )
        return png_bytes
    except Exception as e:
        error_type = extract_validation_error_type(e)
        logger.error(f"SVG to PNG conversion failed ({error_type}): {e}")
        return None


__all__ = [
    "convert_svg_to_png",
]
