"""Language verification node for paper processing.

Verifies that papers are actually in the target language by detecting
language from extracted document content. Papers not in target language
are excluded before the expensive LLM extraction step.
"""

import logging
from collections import defaultdict
from typing import Any

from workflows.shared.language import (
    detect_language,
    extract_detection_sample,
    DEFAULT_CONFIDENCE_THRESHOLD,
)

from .types import PaperProcessingState, LanguageVerificationStats

logger = logging.getLogger(__name__)

# Minimum confidence to accept language detection
LANGUAGE_CONFIDENCE_THRESHOLD = DEFAULT_CONFIDENCE_THRESHOLD


async def language_verification_node(state: PaperProcessingState) -> dict[str, Any]:
    """Verify language of processed papers before extraction.

    For each paper with processing results:
    1. Get content from processing_results (short_summary or ES record)
    2. Combine with title/abstract for detection sample
    3. Detect language and check against target
    4. Remove non-matching papers from papers_to_process

    This node is skipped if:
    - No language_config is set
    - Language is English (no verification needed)

    Returns:
        Updated papers_to_process, language_rejected_dois, language_verification_stats
    """
    language_config = state.get("language_config")
    processing_results = state.get("processing_results", {})
    papers_to_process = state.get("papers_to_process", [])

    # Skip verification for English or if no language config
    if not language_config or language_config.get("code") == "en":
        logger.debug("Skipping language verification (English or no config)")
        return {}

    target_language = language_config["code"]
    target_name = language_config.get("name", target_language)

    logger.info(
        f"Verifying language for {len(processing_results)} processed papers "
        f"(target: {target_name})"
    )

    # Build paper lookup for metadata
    papers_by_doi = {p.get("doi"): p for p in papers_to_process if p.get("doi")}

    verified_dois = []
    rejected_dois = []
    skipped_dois = []
    by_detected_language: dict[str, int] = defaultdict(int)

    for doi, result in processing_results.items():
        paper = papers_by_doi.get(doi, {})
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        # Check if original_language was already detected during document processing
        detected_lang = result.get("original_language")
        confidence = 1.0  # Pre-detected language is trusted

        if detected_lang:
            # Use pre-detected language from document processing
            logger.debug(f"Using pre-detected language for {doi}: {detected_lang}")
        else:
            # Fall back to detection from content
            content = result.get("short_summary", "")

            # If no content available, use metadata only
            if not content and not abstract:
                logger.debug(f"No content for language detection: {doi}")
                skipped_dois.append(doi)
                verified_dois.append(doi)  # Give benefit of doubt
                continue

            # Build detection sample
            sample = extract_detection_sample(
                title=title,
                abstract=abstract,
                content=content,
                max_sample_length=2000,
            )

            # Detect language
            detected_lang, confidence = detect_language(sample)

            if detected_lang is None:
                logger.debug(f"Could not detect language for {doi}")
                skipped_dois.append(doi)
                verified_dois.append(doi)  # Give benefit of doubt
                continue

        by_detected_language[detected_lang] += 1

        if detected_lang == target_language and confidence >= LANGUAGE_CONFIDENCE_THRESHOLD:
            verified_dois.append(doi)
            logger.debug(f"Verified {doi}: {detected_lang} ({confidence:.2f})")
        else:
            rejected_dois.append(doi)
            logger.info(
                f"Rejected {doi}: detected {detected_lang} ({confidence:.2f}), "
                f"expected {target_language}"
            )

    # Filter papers_to_process to only verified DOIs
    verified_papers = [
        p for p in papers_to_process
        if p.get("doi") in verified_dois or p.get("doi") not in processing_results
    ]

    # Apply max_papers limit after verification
    quality_settings = state.get("quality_settings", {})
    max_papers = quality_settings.get("max_papers", 100)

    if len(verified_papers) > max_papers:
        # Sort by relevance score and take top max_papers
        verified_papers = sorted(
            verified_papers,
            key=lambda p: p.get("relevance_score", 0.5),
            reverse=True,
        )[:max_papers]
        logger.info(
            f"Applied max_papers limit: {len(verified_papers)} papers after verification"
        )

    stats: LanguageVerificationStats = {
        "verified_count": len(verified_dois),
        "rejected_count": len(rejected_dois),
        "skipped_count": len(skipped_dois),
        "by_detected_language": dict(by_detected_language),
    }

    logger.info(
        f"Language verification complete: {len(verified_dois)} verified, "
        f"{len(rejected_dois)} rejected, {len(skipped_dois)} skipped"
    )

    if rejected_dois:
        logger.info(f"Language distribution: {dict(by_detected_language)}")

    return {
        "papers_to_process": verified_papers,
        "language_rejected_dois": rejected_dois,
        "language_verification_stats": stats,
    }


def should_verify_language(state: PaperProcessingState) -> str:
    """Router to determine if language verification should run.

    Returns:
        "verify" if verification needed, "skip" otherwise
    """
    language_config = state.get("language_config")

    # Skip if no language config or English
    if not language_config or language_config.get("code") == "en":
        return "skip"

    # Skip if no processing results
    if not state.get("processing_results"):
        return "skip"

    return "verify"
