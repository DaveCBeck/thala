"""Pure document analysis - factual analysis only, no routing decisions.

This module provides document complexity estimation by analyzing PDF characteristics
using PyMuPDF. The analysis is purely factual - routing decisions are made separately
in the routing module.
"""

from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO

import fitz  # PyMuPDF


class DocumentComplexity(Enum):
    """Complexity tiers for document characteristics."""

    LIGHT = "light"  # Text-only, single column
    MIXED = "mixed"  # Some images/tables, manageable layout
    HEAVY = "heavy"  # Image-heavy, complex tables, scanned


@dataclass
class DocumentAnalysis:
    """Factual document analysis result - no routing decisions."""

    complexity: DocumentComplexity
    page_count: int
    has_images: bool
    has_tables: bool
    is_scanned: bool
    avg_image_ratio: float
    multi_column: bool
    multi_column_pages: int
    has_extractable_text: bool


def analyze_document(pdf_content: bytes | BinaryIO) -> DocumentAnalysis:
    """Analyze PDF for factual characteristics.

    Fast pre-flight scan (~50-100ms for 100-page doc).
    Returns pure analysis - routing decisions made separately.

    Args:
        pdf_content: PDF bytes or file-like object

    Returns:
        DocumentAnalysis with complexity tier and characteristics
    """
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    page_count = len(doc)

    total_image_ratio = 0.0
    has_extractable_text = False
    multi_column_pages = 0
    image_pages = 0
    table_likelihood = 0

    for page in doc:
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            continue

        # Check for extractable text (vs scanned)
        text = page.get_text("text")
        if text.strip():
            has_extractable_text = True

        # Image coverage ratio
        images = page.get_images()
        if images:
            image_pages += 1
            image_area = sum(_img_area(page, img) for img in images)
            total_image_ratio += image_area / page_area

        # Multi-column detection
        blocks = page.get_text("dict").get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == 0]
        if text_blocks:
            x_positions = [b["bbox"][0] for b in text_blocks]
            distinct_columns = len(set(round(x, -1) for x in x_positions))
            if distinct_columns > 2:
                multi_column_pages += 1

        # Table heuristic: many small, aligned blocks
        if len(text_blocks) > 20:
            table_likelihood += 1

    doc.close()

    avg_image_ratio = total_image_ratio / page_count if page_count else 0
    is_scanned = not has_extractable_text
    has_images = image_pages > page_count * 0.1 if page_count else False  # >10% pages have images
    has_tables = table_likelihood > page_count * 0.05 if page_count else False  # >5% pages table-like
    multi_column = multi_column_pages > page_count * 0.3 if page_count else False  # >30% multi-column

    # Determine complexity tier (factual, not routing)
    if is_scanned or has_tables or avg_image_ratio > 0.3:
        complexity = DocumentComplexity.HEAVY
    elif has_images or multi_column or avg_image_ratio > 0.1:
        complexity = DocumentComplexity.MIXED
    else:
        complexity = DocumentComplexity.LIGHT

    return DocumentAnalysis(
        complexity=complexity,
        page_count=page_count,
        has_images=has_images,
        has_tables=has_tables,
        is_scanned=is_scanned,
        avg_image_ratio=avg_image_ratio,
        multi_column=multi_column,
        multi_column_pages=multi_column_pages,
        has_extractable_text=has_extractable_text,
    )


def _img_area(page: fitz.Page, img: tuple) -> float:
    """Calculate approximate image area on page."""
    try:
        img_rects = page.get_image_rects(img[0])
        return sum(r.width * r.height for r in img_rects)
    except Exception:
        return 0.0
