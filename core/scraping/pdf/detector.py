"""PDF URL detection utilities."""


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file.

    Handles query parameters and fragments.

    Args:
        url: URL to check

    Returns:
        True if URL appears to be a PDF
    """
    # Strip query params and fragments, check for .pdf extension
    clean_url = url.lower().split("?")[0].split("#")[0].rstrip("/")
    return clean_url.endswith(".pdf")


def validate_pdf_bytes(content: bytes) -> bool:
    """Validate that content is actually a PDF.

    Checks for PDF magic bytes at the start of the file.

    Args:
        content: File content bytes

    Returns:
        True if content appears to be a valid PDF
    """
    return content[:4] == b"%PDF"
