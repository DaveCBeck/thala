"""Text overlap detection for SVG diagrams.

Provides utilities for detecting overlapping text elements in SVG content
using bounding box analysis.
"""

import logging
from typing import NamedTuple

from .schemas import OverlapCheckResult

logger = logging.getLogger(__name__)


class BoundingBox(NamedTuple):
    """Axis-aligned bounding box."""

    x: float
    y: float
    width: float
    height: float

    def overlaps(self, other: "BoundingBox", margin: float = 5.0) -> bool:
        """Check if this box overlaps another (with optional margin)."""
        return not (
            self.x + self.width + margin < other.x
            or other.x + other.width + margin < self.x
            or self.y + self.height + margin < other.y
            or other.y + other.height + margin < self.y
        )


def _estimate_text_bbox(
    text: str,
    x: float,
    y: float,
    font_size: float = 14.0,
    text_anchor: str = "start",
) -> BoundingBox:
    """Estimate bounding box for a text element.

    SVG text elements don't have explicit width/height, so we estimate
    based on character count and font size.
    """
    # Estimate dimensions based on font size
    char_width = font_size * 0.6  # Approximate character width
    width = len(text) * char_width
    height = font_size * 1.2  # Line height

    # Adjust for text-anchor
    if text_anchor == "middle":
        x -= width / 2
    elif text_anchor == "end":
        x -= width

    # Y position in SVG is baseline, so adjust up
    y -= height * 0.8

    return BoundingBox(x=x, y=y, width=width, height=height)


def _parse_font_size(font_size_str: str | None) -> float:
    """Parse font-size attribute to float."""
    if not font_size_str:
        return 14.0
    try:
        # Handle "14px", "14", "1.2em" etc.
        cleaned = font_size_str.replace("px", "").replace("pt", "").replace("em", "")
        return float(cleaned) if cleaned else 14.0
    except ValueError:
        return 14.0


def check_text_overlaps(svg_content: str) -> OverlapCheckResult:
    """Parse SVG and check for text element overlaps.

    Uses lxml to parse SVG and extract text element positions,
    then checks for AABB (Axis-Aligned Bounding Box) overlaps.

    Note: This uses heuristics to avoid false positives from intentional
    multi-line labels (text elements with same x but adjacent y positions).

    Args:
        svg_content: Raw SVG string

    Returns:
        OverlapCheckResult with overlap information
    """
    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml not installed. Run: pip install lxml")
        return OverlapCheckResult(
            has_overlaps=False,
            overlap_pairs=[],
            suggestion="lxml not installed - cannot check overlaps",
        )

    try:
        # Parse SVG
        root = etree.fromstring(svg_content.encode())
    except etree.XMLSyntaxError as e:
        logger.error(f"Invalid SVG: {e}")
        return OverlapCheckResult(
            has_overlaps=False,
            overlap_pairs=[],
            suggestion=f"SVG parsing failed: {e}",
        )

    # Find all text elements (handle SVG namespace)
    namespaces = {"svg": "http://www.w3.org/2000/svg"}

    # Try both namespaced and non-namespaced queries
    text_elements = root.xpath("//svg:text", namespaces=namespaces)
    if not text_elements:
        text_elements = root.xpath("//text")

    # Build bounding boxes for text elements with position info
    boxes: list[tuple[str, BoundingBox, float, float]] = []  # text, bbox, x, y

    for elem in text_elements:
        text = elem.text or ""
        # Also collect text from child tspans
        for child in elem:
            if child.text:
                text += " " + child.text

        text = text.strip()
        if not text:
            continue

        try:
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
        except (ValueError, TypeError):
            continue

        font_size = _parse_font_size(elem.get("font-size"))
        text_anchor = elem.get("text-anchor", "start")

        bbox = _estimate_text_bbox(text, x, y, font_size, text_anchor)
        boxes.append((text[:30], bbox, x, y))  # Truncate long text for reporting

    # Check all pairs for overlaps, but filter out intentional multi-line labels
    overlap_pairs: list[tuple[str, str]] = []
    for i, (text1, box1, x1, y1) in enumerate(boxes):
        for text2, box2, x2, y2 in boxes[i + 1 :]:
            if box1.overlaps(box2, margin=5.0):
                # Heuristic: if x positions are very close (within 15px) and
                # y positions differ by roughly one line height (10-25px),
                # this is likely an intentional multi-line label, not an error
                x_diff = abs(x1 - x2)
                y_diff = abs(y1 - y2)

                # Skip if this looks like intentional stacked text
                if x_diff < 15 and 10 <= y_diff <= 30:
                    continue

                overlap_pairs.append((text1, text2))

    suggestion = None
    if overlap_pairs:
        overlap_desc = "; ".join(
            [f'"{t1}" overlaps "{t2}"' for t1, t2 in overlap_pairs[:5]]
        )
        if len(overlap_pairs) > 5:
            overlap_desc += f" (and {len(overlap_pairs) - 5} more)"
        suggestion = f"Found {len(overlap_pairs)} overlapping text pairs: {overlap_desc}"

    return OverlapCheckResult(
        has_overlaps=bool(overlap_pairs),
        overlap_pairs=overlap_pairs,
        suggestion=suggestion,
    )


__all__ = [
    "BoundingBox",
    "check_text_overlaps",
]
