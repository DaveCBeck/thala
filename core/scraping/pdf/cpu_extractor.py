"""CPU-based PDF extraction for text-heavy documents.

Uses proper lifecycle management for ThreadPoolExecutor to avoid resource leaks.
"""

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Self

import fitz

fitz.TOOLS.mupdf_display_errors(False)


@dataclass
class ExtractionResult:
    """Result from CPU text extraction."""

    markdown: str
    page_count: int
    confidence: float
    fallback_recommended: bool  # True if GPU reprocessing suggested


class CpuExtractor:
    """CPU-based PDF extractor with proper lifecycle management.

    Usage:
        async with CpuExtractor(max_workers=5) as extractor:
            result = await extractor.extract(pdf_content)

    Or for module-level singleton (with explicit shutdown):
        extractor = CpuExtractor(max_workers=5)
        result = await extractor.extract(pdf_content)
        await extractor.shutdown()
    """

    def __init__(self, max_workers: int = 5):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="pdf_cpu_",
        )
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args) -> None:
        await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown executor gracefully."""
        if not self._closed:
            self._executor.shutdown(wait=False)
            self._closed = True

    async def extract(self, pdf_content: bytes) -> ExtractionResult:
        """Extract text from PDF using PyMuPDF (CPU-only).

        Fast path for born-digital, text-heavy documents.
        Returns structured markdown with basic formatting.
        """
        if self._closed:
            raise RuntimeError("Extractor has been shut down")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _extract_sync, pdf_content)


# Module-level singleton for convenience (created lazily)
_default_extractor: CpuExtractor | None = None


def get_extractor() -> CpuExtractor:
    """Get or create default extractor singleton."""
    global _default_extractor
    if _default_extractor is None:
        # Use more workers to saturate multi-core CPUs (e.g., Ryzen 5950X)
        _default_extractor = CpuExtractor(max_workers=16)
    return _default_extractor


async def extract_text_cpu(pdf_content: bytes) -> ExtractionResult:
    """Convenience function using default extractor."""
    return await get_extractor().extract(pdf_content)


def _cleanup_extractor():
    """Shutdown executor gracefully on process exit."""
    if _default_extractor is not None:
        _default_extractor._executor.shutdown(wait=True)


atexit.register(_cleanup_extractor)


def _extract_sync(pdf_content: bytes) -> ExtractionResult:
    """Synchronous extraction in thread pool."""
    doc = fitz.open(stream=pdf_content, filetype="pdf")

    pages = []
    issues = 0

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0]

        # Check for layout complexity
        if text_blocks:
            x_positions = [b["bbox"][0] for b in text_blocks]
            distinct_columns = len(set(round(x, -1) for x in x_positions))
            if distinct_columns > 2:
                issues += 1

        # Extract text with basic structure
        page_text = _format_page_blocks(text_blocks)
        if page_text.strip():
            pages.append(f"<!-- Page {page_num} -->\n\n{page_text}")

    doc.close()

    markdown = "\n\n---\n\n".join(pages)
    confidence = 1.0 - (issues / max(len(pages), 1))

    return ExtractionResult(
        markdown=markdown,
        page_count=len(pages),
        confidence=confidence,
        fallback_recommended=confidence < 0.85,
    )


def _format_page_blocks(blocks: list[dict]) -> str:
    """Format text blocks into basic markdown."""
    lines = []

    for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        block_text = ""
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "")
                flags = span.get("flags", 0)

                # Bold detection (bit 4)
                if flags & 16:
                    text = f"**{text}**"
                # Italic detection (bit 1)
                if flags & 2:
                    text = f"*{text}*"

                line_text += text
            block_text += line_text + "\n"

        lines.append(block_text.strip())

    return "\n\n".join(lines)
