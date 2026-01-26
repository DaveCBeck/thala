"""Text overlap detection for SVG diagrams.

Provides utilities for detecting overlapping text elements in SVG content
using bounding box analysis, as well as bounds violation detection.
"""

import logging
from typing import NamedTuple

from .schemas import BoundsCheckResult, OverlapCheckResult

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


def check_bounds_violations(svg_content: str) -> "BoundsCheckResult":
    """Check if any text or shape elements exceed or are near SVG bounds.

    Detects elements that are cut off at edges or have insufficient margin,
    which makes them illegible or unprofessional.

    Args:
        svg_content: Raw SVG string

    Returns:
        BoundsCheckResult with violation information
    """
    from .schemas import BoundsCheckResult

    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml not installed")
        return BoundsCheckResult(has_violations=False, violations=[])

    try:
        root = etree.fromstring(svg_content.encode())
    except etree.XMLSyntaxError as e:
        logger.error(f"Invalid SVG: {e}")
        return BoundsCheckResult(has_violations=False, violations=[])

    # Get SVG dimensions from viewBox or width/height attributes
    viewbox = root.get("viewBox")
    if viewbox:
        parts = viewbox.split()
        if len(parts) >= 4:
            svg_width = float(parts[2])
            svg_height = float(parts[3])
        else:
            svg_width = float(root.get("width", 800))
            svg_height = float(root.get("height", 600))
    else:
        svg_width = float(root.get("width", "800").replace("px", ""))
        svg_height = float(root.get("height", "600").replace("px", ""))

    violations: list[str] = []
    min_margin = 15  # Minimum margin from edges

    namespaces = {"svg": "http://www.w3.org/2000/svg"}

    # Check text elements
    text_elements = root.xpath("//svg:text", namespaces=namespaces)
    if not text_elements:
        text_elements = root.xpath("//text")

    for elem in text_elements:
        text = elem.text or ""
        for child in elem:
            if child.text:
                text += " " + child.text
        text = text.strip()[:25]  # Truncate for reporting
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

        # Check for bounds violations
        if bbox.x < min_margin:
            violations.append(f'Text "{text}" too close to left edge (x={bbox.x:.0f})')
        if bbox.x + bbox.width > svg_width - min_margin:
            violations.append(
                f'Text "{text}" exceeds right edge '
                f"(extends to {bbox.x + bbox.width:.0f}, max={svg_width - min_margin:.0f})"
            )
        if bbox.y < min_margin:
            violations.append(f'Text "{text}" too close to top edge (y={bbox.y:.0f})')
        if bbox.y + bbox.height > svg_height - min_margin:
            violations.append(
                f'Text "{text}" exceeds bottom edge '
                f"(extends to {bbox.y + bbox.height:.0f}, max={svg_height - min_margin:.0f})"
            )

    return BoundsCheckResult(
        has_violations=bool(violations),
        violations=violations,
        svg_width=svg_width,
        svg_height=svg_height,
    )


def check_text_shape_overlaps(svg_content: str) -> list[str]:
    """Check if text elements overlap with shapes like circles or dots.

    This detects the common issue where data points or decorative circles
    are placed over text labels, making them illegible.

    Args:
        svg_content: Raw SVG string

    Returns:
        List of descriptions of text-shape overlaps found
    """
    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml not installed")
        return []

    try:
        root = etree.fromstring(svg_content.encode())
    except etree.XMLSyntaxError:
        return []

    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    overlaps: list[str] = []

    # Get all text bounding boxes
    text_elements = root.xpath("//svg:text", namespaces=namespaces)
    if not text_elements:
        text_elements = root.xpath("//text")

    text_boxes: list[tuple[str, BoundingBox]] = []
    for elem in text_elements:
        text = elem.text or ""
        for child in elem:
            if child.text:
                text += " " + child.text
        text = text.strip()[:25]
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
        text_boxes.append((text, bbox))

    # Get all circles
    circles = root.xpath("//svg:circle", namespaces=namespaces)
    if not circles:
        circles = root.xpath("//circle")

    circle_boxes: list[BoundingBox] = []
    for elem in circles:
        try:
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            r = float(elem.get("r", 5))
        except (ValueError, TypeError):
            continue
        # Convert circle to bounding box
        circle_boxes.append(BoundingBox(x=cx - r, y=cy - r, width=2 * r, height=2 * r))

    # Get all ellipses
    ellipses = root.xpath("//svg:ellipse", namespaces=namespaces)
    if not ellipses:
        ellipses = root.xpath("//ellipse")

    for elem in ellipses:
        try:
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            rx = float(elem.get("rx", 5))
            ry = float(elem.get("ry", 5))
        except (ValueError, TypeError):
            continue
        circle_boxes.append(BoundingBox(x=cx - rx, y=cy - ry, width=2 * rx, height=2 * ry))

    # Check for overlaps between text and circles/ellipses
    for text, text_bbox in text_boxes:
        for shape_bbox in circle_boxes:
            if text_bbox.overlaps(shape_bbox, margin=2.0):
                overlaps.append(f'Text "{text}" overlapped by circle/dot')
                break  # Only report each text once

    return overlaps


__all__ = [
    "BoundingBox",
    "check_text_overlaps",
    "check_bounds_violations",
    "check_text_shape_overlaps",
]
