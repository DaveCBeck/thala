"""Content classification using structured output abstraction.

Classifies scraped web content to determine:
- full_text: Complete article ready for processing
- abstract_with_pdf: Abstract page with PDF download link
- paywall: Access restricted, needs fallback to retrieve-academic
- non_academic: Not academic content

Uses get_structured_output() for provider-agnostic structured extraction.
"""

import logging
import re
from typing import Optional

from langsmith import traceable

from workflows.shared.llm_utils import ModelTier, get_structured_output
from .prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE
from .types import ClassificationResult

logger = logging.getLogger(__name__)


def _quick_paywall_check(markdown: str) -> bool:
    """Quick heuristic check for obvious paywalls."""
    markdown_lower = markdown.lower()
    paywall_indicators = [
        "sign in to access",
        "sign in to view",
        "subscribe to read",
        "purchase this article",
        "institutional access required",
        "access denied",
        "you do not have access",
        "rent or purchase",
        "buy this article",
        "get full access",
        "login required",
        "members only",
    ]
    matches = sum(1 for indicator in paywall_indicators if indicator in markdown_lower)
    return matches >= 1


def _is_doi_error_page(markdown: str) -> bool:
    """Check if content is a DOI resolver error page."""
    markdown_lower = markdown.lower()
    # DOI error pages are typically short and contain specific error messages
    if len(markdown) > 5000:
        return False  # Error pages are short

    error_indicators = [
        "doi not found",
        "the doi system",
        "doi.org",
        "handle not found",
        "invalid doi",
        "doi could not be resolved",
        "resource not found",
        "the requested doi",
        "doi resolution failed",
    ]
    matches = sum(1 for indicator in error_indicators if indicator in markdown_lower)
    return matches >= 2  # Need at least 2 indicators


def _has_article_structure(markdown: str) -> bool:
    """Check if markdown has typical article section headers."""
    section_patterns = [
        r"#+\s*introduction",
        r"#+\s*methods",
        r"#+\s*materials?\s*(and|&)\s*methods?",
        r"#+\s*results",
        r"#+\s*discussion",
        r"#+\s*conclusion",
        r"#+\s*background",
        r"#+\s*abstract",
        r"#+\s*references",
    ]
    matches = sum(1 for p in section_patterns if re.search(p, markdown, re.IGNORECASE))
    return matches >= 3  # At least 3 typical sections


@traceable(name="classify_content", run_type="llm")
async def classify_content(
    url: str,
    markdown: str,
    links: list[str],
    doi: Optional[str] = None,
) -> ClassificationResult:
    """Classify scraped content using structured output.

    Uses get_structured_output() for provider-agnostic structured extraction.
    Applies quick heuristics first for obvious cases.

    Args:
        url: Original URL that was scraped
        markdown: Scraped markdown content
        links: List of links found on the page
        doi: DOI if known (for context)

    Returns:
        ClassificationResult with classification and optional PDF URL
    """
    # Fast path: heuristic detection for obvious cases
    if _is_doi_error_page(markdown):
        logger.debug("Quick DOI error page detection")
        return ClassificationResult(
            classification="paywall",  # Treat as paywall to trigger fallback
            confidence=0.95,
            pdf_url=None,
            reasoning="Content is a DOI resolver error page (DOI not found)",
        )

    if _quick_paywall_check(markdown):
        logger.debug("Quick paywall detection")
        return ClassificationResult(
            classification="paywall",
            confidence=0.95,
            pdf_url=None,
            reasoning="Content contains clear paywall/access restriction indicators",
        )

    if len(markdown) > 20000 and _has_article_structure(markdown):
        logger.debug(f"Quick full_text detection: {len(markdown)} chars with structure")
        return ClassificationResult(
            classification="full_text",
            confidence=0.9,
            pdf_url=None,
            reasoning="Long content (>20k chars) with typical article section structure",
        )

    # LLM classification for ambiguous cases
    logger.debug("Classifying content via DeepSeek V3")

    # Truncate content for classification
    content_preview = markdown[:15000] if len(markdown) > 15000 else markdown
    links_text = (
        "\n".join(f"- {link}" for link in links[:30]) if links else "(no links found)"
    )

    user_prompt = CLASSIFICATION_USER_TEMPLATE.format(
        url=url,
        doi=doi or "unknown",
        content_length=len(markdown),
        content_preview=content_preview,
        links_text=links_text,
    )

    try:
        result: ClassificationResult = await get_structured_output(
            output_schema=ClassificationResult,
            user_prompt=user_prompt,
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            tier=ModelTier.DEEPSEEK_V3,
            max_tokens=1024,
        )

        logger.debug(
            f"Classification: {result.classification} "
            f"(confidence={result.confidence:.2f}, pdf_url={result.pdf_url is not None})"
        )
        return result

    except Exception as e:
        logger.error(f"Classification failed: {type(e).__name__}: {e}")
        # Default to full_text on error to avoid losing content
        return ClassificationResult(
            classification="full_text",
            confidence=0.5,
            pdf_url=None,
            reasoning=f"Classification error, defaulting to full_text: {e}",
        )
