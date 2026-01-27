"""PDF URL detection utilities."""

# Common PDF URL patterns used by academic publishers
# These patterns indicate URLs that serve PDF content even without .pdf extension
PDF_PATH_PATTERNS = [
    "/pdfdirect/",  # Wiley, AGU
    "/doi/pdf/",  # Royal Society, various publishers
    "/doi/pdfdirect/",  # Some Wiley subdomains
    "/content/pdf/",  # Springer, Nature
    "/article/pdf/",  # ScienceDirect
    "/pdf/",  # Generic PDF path segment
]


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file.

    Handles:
    - URLs ending in .pdf (with query params/fragments stripped)
    - URLs with common academic publisher PDF path patterns
      (e.g., /pdfdirect/, /doi/pdf/, /content/pdf/)

    Args:
        url: URL to check

    Returns:
        True if URL appears to be a PDF
    """
    # Strip query params and fragments for extension check
    clean_url = url.lower().split("?")[0].split("#")[0].rstrip("/")

    # Check for .pdf extension
    if clean_url.endswith(".pdf"):
        return True

    # Check for common PDF URL patterns (case-insensitive)
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in PDF_PATH_PATTERNS)


def validate_pdf_bytes(content: bytes) -> bool:
    """Validate that content is actually a PDF.

    Checks for PDF magic bytes at the start of the file.

    Args:
        content: File content bytes

    Returns:
        True if content appears to be a valid PDF
    """
    return content[:4] == b"%PDF"
