"""
Core processing nodes for document processing workflow.
"""

from .chapter_detector import detect_chapters
from .finalizer import finalize
from .input_resolver import resolve_input
from .language_detector import detect_document_language
from .metadata_agent import check_metadata
from .save_short_summary import save_short_summary
from .save_tenth_summary import save_tenth_summary
from .store_updater import update_store
from .summary_agent import generate_summary
from .update_zotero import update_zotero
from .zotero_stub import create_zotero_stub

__all__ = [
    "resolve_input",
    "create_zotero_stub",
    "update_store",
    "detect_document_language",
    "generate_summary",
    "check_metadata",
    "save_short_summary",
    "detect_chapters",
    "save_tenth_summary",
    "update_zotero",
    "finalize",
]
