"""Language detection utilities using langdetect.

Provides language detection for verifying paper content matches expected language.
Uses the same library (langdetect) that OpenAlex uses internally.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum text length for reliable detection
MIN_TEXT_LENGTH = 50
# Default confidence threshold for accepting a detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


def detect_language(
    text: str,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> tuple[Optional[str], float]:
    """
    Detect language of text using langdetect.

    Args:
        text: Text to detect language from
        min_text_length: Minimum characters required for detection

    Returns:
        Tuple of (language_code, confidence):
        - language_code: ISO 639-1 code (e.g., "en", "de") or None if detection fails
        - confidence: 0.0-1.0 confidence score (0.0 if detection fails)

    Example:
        >>> lang, conf = detect_language("Dies ist ein deutscher Text")
        >>> print(f"{lang}: {conf:.2f}")
        de: 0.99
    """
    # Import here to avoid import errors if langdetect not installed
    try:
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        logger.error("langdetect not installed. Run: pip install langdetect")
        return None, 0.0

    if not text or len(text.strip()) < min_text_length:
        logger.debug(f"Text too short for language detection ({len(text)} chars)")
        return None, 0.0

    try:
        # detect_langs returns list of Language objects with lang and prob attributes
        results = detect_langs(text)
        if results:
            top_result = results[0]
            return top_result.lang, top_result.prob
        return None, 0.0

    except LangDetectException as e:
        logger.debug(f"Language detection failed: {e}")
        return None, 0.0
    except Exception as e:
        logger.warning(f"Unexpected error in language detection: {e}")
        return None, 0.0


def verify_language_match(
    text: str,
    target_language: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> tuple[bool, str | None, float]:
    """
    Verify that text matches the target language.

    Args:
        text: Text to verify
        target_language: Expected ISO 639-1 language code (e.g., "de", "fr")
        confidence_threshold: Minimum confidence to accept detection
        min_text_length: Minimum text length for detection

    Returns:
        Tuple of (is_match, detected_language, confidence):
        - is_match: True if detected language matches target with sufficient confidence
        - detected_language: What language was actually detected (or None)
        - confidence: Detection confidence score

    Example:
        >>> is_match, detected, conf = verify_language_match(
        ...     "This is English text",
        ...     target_language="de"
        ... )
        >>> print(f"Match: {is_match}, Detected: {detected}")
        Match: False, Detected: en
    """
    detected_lang, confidence = detect_language(text, min_text_length)

    if detected_lang is None:
        # Could not detect - consider it a non-match
        return False, None, 0.0

    if confidence < confidence_threshold:
        # Low confidence - consider it uncertain (non-match)
        logger.debug(
            f"Low confidence detection: {detected_lang} ({confidence:.2f} < {confidence_threshold})"
        )
        return False, detected_lang, confidence

    is_match = detected_lang == target_language
    return is_match, detected_lang, confidence


def extract_detection_sample(
    title: str,
    abstract: str | None = None,
    content: str | None = None,
    max_sample_length: int = 2000,
) -> str:
    """
    Extract a text sample for language detection.

    Combines title, abstract, and content start for best detection accuracy.
    Prioritizes abstract (usually most language-indicative) over raw content.

    Args:
        title: Paper title
        abstract: Paper abstract (if available)
        content: Full paper content (if available)
        max_sample_length: Maximum length of combined sample

    Returns:
        Combined text sample for language detection
    """
    parts = []

    if title:
        parts.append(title)

    if abstract:
        parts.append(abstract)

    # If we have content and need more text, add content start
    current_length = sum(len(p) for p in parts)
    if content and current_length < max_sample_length:
        remaining = max_sample_length - current_length
        # Take first N characters of content
        content_sample = content[:remaining].strip()
        if content_sample:
            parts.append(content_sample)

    return "\n\n".join(parts)
