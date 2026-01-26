"""SVG to PNG conversion utilities.

Provides functions for converting SVG content to PNG using CairoSVG.
"""

import logging

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

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            dpi=dpi,
            background_color=background_color,
        )
        return png_bytes
    except Exception as e:
        logger.error(f"SVG to PNG conversion failed: {e}")
        return None


__all__ = [
    "convert_svg_to_png",
]
