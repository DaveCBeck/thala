---
module: core/scraping
date: 2026-01-14
problem_type: code_debt
component: document_processing
symptoms:
  - "Duplicate processing paths for PDF and EPUB formats"
  - "EPUB processing lacked advanced features (layout detection, tables)"
  - "Inconsistent output quality between PDF and EPUB"
  - "Separate error handling and metrics for similar operations"
root_cause: incremental_development
resolution_type: refactoring
severity: medium
tags: [consolidation, epub, marker, refactoring, simplification, document-processing]
---

# EPUB Processing Consolidation

## Problem

The document processing pipeline had dual paths for similar file formats:

- **PDF files**: Processed through Marker service with advanced features
- **EPUB files**: Processed through standalone `core/scraping/epub.py` (238 lines)

The standalone EPUB processor used basic `html2text` extraction, lacking:
- Layout detection and table handling
- Metadata stripping
- OCR capabilities for embedded images
- Consistent quality with PDF processing

```python
# ❌ BEFORE: Dual processing architecture
User Document
    ├─ PDF file  → process_pdf_file() → Marker → Markdown (high quality)
    └─ EPUB file → process_epub_file() → html2text → Markdown (basic quality)
```

### Impact

- Inconsistent output quality depending on input format
- Code duplication in routing logic
- Different error handling and metrics naming (`epub:*` vs `marker:*`)
- Maintenance burden for separate EPUB module

## Root Cause

**Incremental development**: EPUB support was added separately before discovering Marker already handles EPUB files via its `marker-pdf[full]` package with PyMuPDF.

## Solution

Route all EPUB files through Marker, which auto-detects the format and provides unified processing quality.

```python
# ✅ AFTER: Unified processing architecture
User Document
    ├─ PDF file  → process_pdf_file() → Marker → Markdown
    └─ EPUB file → process_pdf_file() → Marker → Markdown
        (Marker auto-detects EPUB and converts internally)
```

### Code Changes

**input_resolver.py - Before:**
```python
from core.scraping.epub import process_epub_file

EPUB_EXTENSIONS = {".epub"}

elif suffix in EPUB_EXTENSIONS:
    logger.info(f"Processing local EPUB: {source_path.name}")
    try:
        markdown = process_epub_file(str(source_path))
        ocr_method = "epub:extracted"
    except Exception as e:
        logger.error(f"EPUB processing failed: {e}")
        markdown = f"[EPUB processing failed: {source_path.name}]"
        ocr_method = "epub:failed"
```

**input_resolver.py - After:**
```python
from core.scraping.pdf import process_pdf_file

EPUB_EXTENSIONS = {".epub"}

elif suffix in EPUB_EXTENSIONS:
    logger.info(f"Processing local EPUB via Marker: {source_path.name}")
    try:
        markdown = await process_pdf_file(
            str(source_path),
            quality="balanced",
            langs=input_data.get("langs", ["English"]),
        )
        ocr_method = "marker:epub"
    except Exception as e:
        logger.error(f"Marker EPUB processing failed: {e}")
        markdown = f"[EPUB processing failed: {source_path.name}]"
        ocr_method = "marker:epub_failed"
```

### Files Modified

**Removed:**
- `core/scraping/epub.py` (238 lines) - Entire standalone EPUB processor

**Updated:**
- `core/scraping/__init__.py` - Removed EPUB exports
- `workflows/document_processing/nodes/input_resolver.py` - Route EPUB to Marker

### Exports Removed

```python
# REMOVED from core/scraping/__init__.py
from .epub import process_epub_bytes, process_epub_file

__all__ = [
    # "process_epub_bytes",  # REMOVED
    # "process_epub_file",   # REMOVED
]
```

## How Marker Handles EPUB

Marker's internal EPUB processing:

1. **Auto-detection**: PdfConverter detects EPUB format from file content
2. **Conversion**: Uses PyMuPDF (`fitz`) to convert EPUB → PDF internally
3. **Processing**: Standard Marker pipeline with layout detection, tables, metadata stripping
4. **Output**: High-quality markdown matching PDF output quality

**Dependency**: Marker service requires `marker-pdf[full]` package (includes PyMuPDF).

## Benefits

| Benefit | Impact |
|---------|--------|
| **Unified quality** | EPUB and PDF now have identical processing quality |
| **Code reduction** | 252 lines removed (238 module + 14 routing) |
| **Single pipeline** | One `process_pdf_file()` handles all document formats |
| **Consistent metrics** | `marker:epub` naming matches `marker:pdf` convention |
| **Advanced features** | EPUB now gets layout detection, table handling, OCR |
| **Simpler maintenance** | No separate EPUB logic to maintain |

## Prevention

When adding new format support:

1. **Check existing services first**: Marker supports PDF, DOCX, EPUB, PPTX, MOBI, and more
2. **Prefer unified pipelines**: Route through existing services when quality is acceptable
3. **Document format support**: Update service documentation with supported formats

## Related Patterns

- [GPU-Accelerated Document Processing](../../patterns/data-pipeline/gpu-accelerated-document-processing.md) - Marker integration details
- [Unified Content Retrieval Pipeline](../../patterns/data-pipeline/unified-content-retrieval-pipeline.md) - Pipeline architecture
- [File Format Detection and Validation](../../patterns/data-pipeline/file-format-detection-validation.md) - Format detection

## Related Solutions

- [Systematic Deprecation Migration](./systematic-deprecation-migration.md) - Code consolidation patterns

## References

- [Marker PDF Documentation](https://github.com/VikParuchuri/marker)
- [PyMuPDF EPUB Support](https://pymupdf.readthedocs.io/)
