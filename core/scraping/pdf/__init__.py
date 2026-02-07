"""PDF detection and processing subpackage."""

from .analysis import DocumentAnalysis, DocumentComplexity, analyze_document
from .detector import is_pdf_url
from .processor import (
    check_marker_available,
    download_pdf_by_md5,
    MarkerProcessingError,
    process_pdf_by_md5,
    process_pdf_bytes,
    process_pdf_file,
    process_pdf_url,
)
from .router import (
    check_marker_for_session,
    ProcessingResult,
    process_document_smart,
    process_document_smart_url,
    reset_marker_session_cache,
)
from .routing import RouteDecision, determine_route

__all__ = [
    # Document analysis
    "analyze_document",
    "DocumentAnalysis",
    "DocumentComplexity",
    # Routing
    "determine_route",
    "RouteDecision",
    # Smart processing (recommended entry point)
    "check_marker_for_session",
    "process_document_smart",
    "process_document_smart_url",
    "ProcessingResult",
    "reset_marker_session_cache",
    # Legacy/direct Marker processing
    "check_marker_available",
    "is_pdf_url",
    "download_pdf_by_md5",
    "MarkerProcessingError",
    "process_pdf_by_md5",
    "process_pdf_bytes",
    "process_pdf_file",
    "process_pdf_url",
]
