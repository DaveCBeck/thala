"""
Language detection node for document processing workflow.

Detects the original language from L0 content using langdetect.
Updates L0 record with detected language_code.
"""

import logging
from typing import Any
from uuid import UUID

from langchain_tools.base import get_store_manager

from workflows.document_processing.state import DocumentProcessingState
from workflows.shared.language.detection import detect_language

logger = logging.getLogger(__name__)

# Use first 5000 chars for detection (enough for accuracy, fast processing)
MAX_DETECTION_SAMPLE = 5000


async def detect_document_language(state: DocumentProcessingState) -> dict[str, Any]:
    """
    Detect original language from L0 content.

    Uses first ~5000 chars of markdown for detection.
    Updates L0 record with language_code.
    Returns original_language (ISO 639-1) and confidence.
    """
    try:
        processing_result = state.get("processing_result")
        if not processing_result:
            logger.error("No processing_result in state")
            return {
                "original_language": "en",  # Default to English
                "original_language_confidence": 0.0,
                "errors": [{"node": "language_detector", "error": "No processing result"}],
            }

        markdown = processing_result["markdown"]

        # Get detection sample from content
        sample = markdown[:MAX_DETECTION_SAMPLE]

        # Detect language
        detected_lang, confidence = detect_language(sample)

        if detected_lang is None:
            logger.warning("Could not detect language, defaulting to 'en'")
            return {
                "original_language": "en",
                "original_language_confidence": 0.0,
            }

        logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2f})")

        # Update L0 record with detected language
        store_records = state.get("store_records", [])
        if store_records:
            record_id = UUID(store_records[0]["id"])
            store_manager = get_store_manager()
            await store_manager.es_stores.store.update(
                record_id,
                {"language_code": detected_lang},
                compression_level=0,
            )
            logger.debug(f"Updated L0 record {record_id} with language_code={detected_lang}")

        return {
            "original_language": detected_lang,
            "original_language_confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return {
            "original_language": "en",  # Default to English on error
            "original_language_confidence": 0.0,
            "errors": [{"node": "language_detector", "error": str(e)}],
        }
