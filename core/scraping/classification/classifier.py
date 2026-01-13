"""Content classification using direct Anthropic SDK.

Classifies scraped web content to determine:
- full_text: Complete article ready for processing
- abstract_with_pdf: Abstract page with PDF download link
- paywall: Access restricted, needs fallback to retrieve-academic
- non_academic: Not academic content
"""

import logging
import os
import re
from typing import Optional

import anthropic

from .prompts import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE
from .types import ClassificationResult

logger = logging.getLogger(__name__)

# Use Haiku for fast, cheap classification
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"


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


async def classify_content(
    url: str,
    markdown: str,
    links: list[str],
    doi: Optional[str] = None,
) -> ClassificationResult:
    """Classify scraped content using Anthropic Haiku.

    Uses direct Anthropic SDK with tool_use for structured output.
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
    if _quick_paywall_check(markdown):
        logger.debug(f"Quick paywall detection for {url}")
        return ClassificationResult(
            classification="paywall",
            confidence=0.95,
            pdf_url=None,
            reasoning="Content contains clear paywall/access restriction indicators",
        )

    if len(markdown) > 20000 and _has_article_structure(markdown):
        logger.debug(f"Quick full_text detection for {url}: {len(markdown)} chars with structure")
        return ClassificationResult(
            classification="full_text",
            confidence=0.9,
            pdf_url=None,
            reasoning="Long content (>20k chars) with typical article section structure",
        )

    # LLM classification for ambiguous cases
    logger.info(f"Classifying content via Haiku: {url}")

    # Truncate content for classification
    content_preview = markdown[:15000] if len(markdown) > 15000 else markdown
    links_text = "\n".join(f"- {link}" for link in links[:30]) if links else "(no links found)"

    user_prompt = CLASSIFICATION_USER_TEMPLATE.format(
        url=url,
        doi=doi or "unknown",
        content_length=len(markdown),
        content_preview=content_preview,
        links_text=links_text,
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = await client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=1024,
            system=CLASSIFICATION_SYSTEM_PROMPT,
            tools=[
                {
                    "name": "classify_content",
                    "description": "Classify academic content type and extract PDF URL if applicable",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "classification": {
                                "type": "string",
                                "enum": ["full_text", "abstract_with_pdf", "paywall", "non_academic"],
                                "description": "The classification of the content",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score between 0 and 1",
                            },
                            "pdf_url": {
                                "type": "string",
                                "description": "URL to download PDF if classification is abstract_with_pdf. Must be a valid HTTP/HTTPS URL.",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of the classification decision",
                            },
                            "title": {
                                "type": "string",
                                "description": "Article title if this is academic content",
                            },
                            "authors": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of author names if this is academic content",
                            },
                        },
                        "required": ["classification", "confidence", "reasoning"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "classify_content"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract tool call result
        for block in response.content:
            if block.type == "tool_use" and block.name == "classify_content":
                result = ClassificationResult(**block.input)
                logger.info(
                    f"Classification for {url}: {result.classification} "
                    f"(confidence={result.confidence:.2f}, pdf_url={result.pdf_url is not None})"
                )
                return result

        # Fallback if no tool call in response
        logger.warning(f"No tool call in classification response for {url}")
        return ClassificationResult(
            classification="full_text",
            confidence=0.5,
            pdf_url=None,
            reasoning="Fallback: no tool call in LLM response",
        )

    except Exception as e:
        logger.error(f"Classification failed for {url}: {type(e).__name__}: {e}")
        # Default to full_text on error to avoid losing content
        return ClassificationResult(
            classification="full_text",
            confidence=0.5,
            pdf_url=None,
            reasoning=f"Classification error, defaulting to full_text: {e}",
        )
