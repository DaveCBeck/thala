---
name: file-format-detection-validation
title: "File Format Detection and Content Validation"
date: 2026-01-14
category: data-pipeline
applicability:
  - "Document retrieval pipelines processing PDFs, EPUBs, and other formats"
  - "Systems downloading content from untrusted sources"
  - "Pipelines needing to detect HTML error pages masquerading as documents"
  - "Multi-format processing with automatic routing"
components: [magic_bytes, format_validator, extension_corrector, validation_result]
complexity: moderate
verified_in_production: true
related_solutions:
  - scraping-pdf-robustness-fixes
  - html-content-classification
tags: [validation, magic-bytes, pdf, epub, format-detection, content-validation, file-processing]
---

# File Format Detection and Content Validation

## Intent

Detect actual file format from content bytes (not headers or extensions) and validate content integrity before expensive processing.

## Motivation

When downloading documents from various sources, the actual file format often doesn't match expectations:

- **Spoofed headers**: Server sends `Content-Type: application/pdf` but serves HTML paywall
- **Wrong extensions**: File named `.pdf` but is actually EPUB or HTML
- **Stub documents**: Valid PDF structure but only preview/abstract pages
- **Error pages**: HTML error or login pages saved with document extensions

These mismatches cause processing failures, wasted compute, and corrupted outputs. Validating at the byte level catches issues before expensive operations (GPU processing, embedding generation).

## Applicability

Use this pattern when:
- Downloading documents from external sources (publishers, repositories)
- Processing multiple file formats through a unified pipeline
- Needing to reject incomplete or corrupt documents early
- Building robust document retrieval with fallback chains

Do NOT use this pattern when:
- Processing only trusted, pre-validated content
- File format is guaranteed by upstream systems
- Speed is critical and validation overhead unacceptable

## Structure

```
Download Content
       │
       ▼
┌─────────────────────────┐
│  Magic Bytes Detection  │  PDF: %PDF, ZIP: PK\x03\x04
│    detect_format()      │  Immune to header/extension spoofing
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Format-Specific        │  PDF: page count, content density
│  Validation             │  EPUB: container.xml, content files
│    validate_content()   │  HTML: detect error pages
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Extension Correction   │  Rename .pdf → .epub if detected
│  (Optional)             │  Preserve debug copies
└───────────┬─────────────┘
            │
            ▼
    ValidationResult(
      detected_format,
      is_valid,
      page_count,
      error
    )
```

## Implementation

### Step 1: Magic Byte Detection

Detect format from first few bytes, immune to header spoofing:

```python
# Magic byte signatures
PDF_MAGIC = b"%PDF"
ZIP_MAGIC = b"PK\x03\x04"  # EPUB is ZIP with specific structure


def detect_format_from_content(content: bytes) -> str:
    """Detect file format from magic bytes.

    Returns: 'pdf', 'epub', 'zip', 'html', or 'unknown'
    """
    if len(content) < 4:
        return "unknown"

    # Check PDF magic bytes
    if content[:4] == PDF_MAGIC:
        return "pdf"

    # Check ZIP magic (EPUB is ZIP with specific structure)
    if content[:4] == ZIP_MAGIC:
        return _detect_zip_subtype(content)

    # Check for HTML (error pages)
    if b"<!DOCTYPE html" in content[:100] or b"<html" in content[:100]:
        return "html"

    return "unknown"


def _detect_zip_subtype(content: bytes) -> str:
    """Distinguish EPUB from generic ZIP."""
    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            namelist = zf.namelist()

            # EPUB has META-INF/container.xml or mimetype file
            if "META-INF/container.xml" in namelist:
                return "epub"
            if "mimetype" in namelist:
                mimetype = zf.read("mimetype").decode("utf-8", errors="ignore")
                if "epub" in mimetype.lower():
                    return "epub"
            return "zip"
    except zipfile.BadZipFile:
        return "unknown"
```

### Step 2: Format-Specific Validation

Validate content integrity based on detected format:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """Result of file validation."""
    detected_format: str
    is_valid: bool
    page_count: Optional[int] = None
    error: Optional[str] = None


# Validation thresholds
MIN_PDF_FILE_SIZE = 10 * 1024  # 10 KB
MIN_PDF_WORDS_SINGLE_PAGE = 300  # Reject stub/preview PDFs


def validate_pdf_content(content: bytes) -> ValidationResult:
    """Validate PDF for page count and content density."""

    # Size check
    if len(content) < MIN_PDF_FILE_SIZE:
        return ValidationResult(
            detected_format="pdf",
            is_valid=False,
            error=f"PDF too small ({len(content)} bytes), likely corrupt",
        )

    # Parse with pypdf
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(content))
        page_count = len(reader.pages)

        if page_count == 0:
            return ValidationResult(
                detected_format="pdf",
                is_valid=False,
                page_count=0,
                error="PDF has no pages",
            )

        # Content density check for single-page PDFs
        if page_count == 1:
            text = reader.pages[0].extract_text() or ""
            word_count = len(text.split())
            if word_count < MIN_PDF_WORDS_SINGLE_PAGE:
                return ValidationResult(
                    detected_format="pdf",
                    is_valid=False,
                    page_count=1,
                    error=f"PDF appears to be stub ({word_count} words)",
                )

        return ValidationResult(
            detected_format="pdf",
            is_valid=True,
            page_count=page_count,
        )

    except Exception as e:
        return ValidationResult(
            detected_format="pdf",
            is_valid=False,
            error=f"Failed to parse PDF: {e}",
        )


def validate_epub_content(content: bytes) -> ValidationResult:
    """Validate EPUB structure and content presence."""
    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            namelist = zf.namelist()

            # Check EPUB structure
            has_container = "META-INF/container.xml" in namelist
            has_mimetype = "mimetype" in namelist

            if not (has_container or has_mimetype):
                return ValidationResult(
                    detected_format="epub",
                    is_valid=False,
                    error="Missing META-INF/container.xml or mimetype",
                )

            # Check for content files
            content_files = [
                n for n in namelist
                if n.endswith((".html", ".xhtml", ".htm"))
            ]

            if not content_files:
                return ValidationResult(
                    detected_format="epub",
                    is_valid=False,
                    error="EPUB has no content files",
                )

            return ValidationResult(
                detected_format="epub",
                is_valid=True,
            )

    except zipfile.BadZipFile as e:
        return ValidationResult(
            detected_format="epub",
            is_valid=False,
            error=f"Corrupt EPUB (invalid ZIP): {e}",
        )
```

### Step 3: Unified Validation Entry Point

```python
def validate_file(
    content: bytes,
    expected_format: str = "pdf",
) -> ValidationResult:
    """Validate file content and detect actual format.

    Args:
        content: Raw file bytes
        expected_format: Expected format based on URL/headers

    Returns:
        ValidationResult with detected format and validity
    """
    detected_format = detect_format_from_content(content)

    # Log mismatches
    if detected_format != expected_format and detected_format != "unknown":
        logger.warning(
            f"Format mismatch: expected {expected_format}, "
            f"detected {detected_format}"
        )

    # Route to format-specific validator
    if detected_format == "pdf":
        return validate_pdf_content(content)
    elif detected_format == "epub":
        return validate_epub_content(content)
    elif detected_format == "html":
        return ValidationResult(
            detected_format="html",
            is_valid=False,
            error="Downloaded HTML instead of document (likely paywall)",
        )
    else:
        return ValidationResult(
            detected_format="unknown",
            is_valid=False,
            error=f"Unknown format (expected {expected_format})",
        )
```

### Step 4: Extension Correction

```python
from pathlib import Path


def correct_file_extension(
    file_path: Path,
    detected_format: str,
) -> Path:
    """Rename file if extension doesn't match detected format.

    Only corrects for known formats (pdf, epub, txt).
    """
    current_ext = file_path.suffix.lower().lstrip(".")

    if current_ext == detected_format:
        return file_path

    if detected_format not in ("pdf", "epub", "txt"):
        return file_path

    new_path = file_path.with_suffix(f".{detected_format}")

    if file_path.exists():
        logger.info(f"Correcting extension: {file_path.name} → {new_path.name}")
        file_path.rename(new_path)

    return new_path
```

## Complete Example

Integration in document download workflow:

```python
async def download_and_validate(
    url: str,
    doi: str,
    download_dir: Path,
    source: str = "unpaywall",
) -> tuple[Path, ValidationResult]:
    """Download document and validate content."""

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        content = response.content

    # Determine expected format from URL/headers
    content_type = response.headers.get("content-type", "")
    if "pdf" in content_type.lower() or url.endswith(".pdf"):
        expected = "pdf"
    elif "epub" in content_type.lower() or url.endswith(".epub"):
        expected = "epub"
    else:
        expected = "pdf"

    # Validate content
    validation = validate_file(content, expected_format=expected)

    if not validation.is_valid:
        # Save invalid file for debugging
        debug_path = download_dir / f"{source}_{doi}_invalid.{expected}"
        debug_path.write_bytes(content)
        raise DownloadError(
            f"Validation failed: {validation.error} "
            f"(detected: {validation.detected_format})"
        )

    # Use detected format for extension
    extension = validation.detected_format
    if extension not in ("pdf", "epub", "txt"):
        extension = expected

    file_path = download_dir / f"{source}_{doi}.{extension}"
    file_path.write_bytes(content)

    # Correct extension if needed
    file_path = correct_file_extension(file_path, validation.detected_format)

    logger.info(
        f"Downloaded {file_path.name}: {len(content) / 1024:.1f} KB, "
        f"{validation.page_count or 'N/A'} pages"
    )

    return file_path, validation
```

## Consequences

### Benefits

- **Security**: Rejects HTML paywalls disguised as documents
- **Reliability**: Catches stub/preview PDFs before expensive processing
- **Accuracy**: Corrects mislabeled files automatically
- **Debuggability**: Saves invalid files for investigation
- **Extensibility**: Easy to add new format validators

### Trade-offs

- **Overhead**: Validation adds processing time for every download
- **Dependencies**: Requires pypdf for PDF validation
- **False positives**: Single-page PDFs with legitimate sparse content may be rejected
- **Memory**: Large files fully loaded for validation

### Alternatives

- **Header-only validation**: Faster but vulnerable to spoofing
- **Deferred validation**: Validate during processing (catches issues later)
- **Sampling**: Check first N bytes only (faster but less thorough)

## Related Patterns

- [Unified Content Retrieval Pipeline](./unified-content-retrieval-pipeline.md) - Integration point for format detection
- [GPU-Accelerated Document Processing](./gpu-accelerated-document-processing.md) - Multi-format processing
- [Unified Scraping Service](./unified-scraping-service-fallback-chain.md) - Fallback chains for retrieval

## Related Solutions

- [Scraping and PDF Robustness Fixes](../../solutions/scraping-issues/scraping-pdf-robustness-fixes.md) - PDF detection in scraping
- [HTML Content Classification](../../solutions/workflow-issues/html-content-classification.md) - LLM-based content classification

## Known Uses in Thala

- `services/retrieve-academic/app/retriever/validation.py` - Core validation logic
- `services/retrieve-academic/app/retriever/download.py` - Download integration
- `core/scraping/epub.py` - EPUB content extraction
- `workflows/document_processing/nodes/input_resolver.py` - Local file handling

## References

- [File Signatures (Magic Numbers)](https://en.wikipedia.org/wiki/List_of_file_signatures)
- [pypdf Documentation](https://pypdf.readthedocs.io/)
- [EPUB 3 Specification](https://www.w3.org/publishing/epub32/epub-spec.html)
