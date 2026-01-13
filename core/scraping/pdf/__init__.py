"""PDF detection and processing subpackage."""

from .detector import is_pdf_url
from .processor import (
    process_pdf_by_md5,
    process_pdf_bytes,
    process_pdf_file,
    process_pdf_url,
)

__all__ = [
    "is_pdf_url",
    "process_pdf_by_md5",
    "process_pdf_bytes",
    "process_pdf_file",
    "process_pdf_url",
]
