"""PDF detection and processing subpackage."""

from .analysis import DocumentAnalysis, DocumentComplexity, analyze_document
from .detector import is_pdf_url
from .processor import (
    download_pdf_by_md5,
    MarkerProcessingError,
    process_pdf_by_md5,
    process_pdf_bytes,
    process_pdf_file,
    process_pdf_url,
)
from .router import ProcessingResult, process_document_smart
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
    "process_document_smart",
    "ProcessingResult",
    # Legacy/direct Marker processing
    "is_pdf_url",
    "download_pdf_by_md5",
    "MarkerProcessingError",
    "process_pdf_by_md5",
    "process_pdf_bytes",
    "process_pdf_file",
    "process_pdf_url",
]
