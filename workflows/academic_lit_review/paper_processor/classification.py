"""Content classification for scraped HTML pages.

Uses Haiku for fast, cheap batch classification of whether scraped content is:
- full_text: Complete article ready for processing
- abstract_with_pdf: Abstract page with PDF download link
- paywall: Access restricted, needs fallback to retrieve-academic
"""

import logging
import re
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from workflows.shared.llm_utils import ModelTier, get_structured_output

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schemas for Structured Output
# =============================================================================


class ClassificationItem(BaseModel):
    """Classification result for a single scraped page."""

    doi: str = Field(description="DOI of the paper being classified")
    classification: Literal["full_text", "abstract_with_pdf", "paywall"] = Field(
        description="Type of content detected"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the classification (0.0-1.0)",
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="URL to PDF if classification is abstract_with_pdf. Extract from the links list.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen",
    )

    @field_validator("pdf_url", mode="before")
    @classmethod
    def validate_pdf_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate that pdf_url is actually a URL, not page content."""
        if v is None:
            return None
        # Quick sanity checks
        if len(v) > 2000:  # URLs shouldn't be this long
            logger.warning(f"Rejecting pdf_url: too long ({len(v)} chars)")
            return None
        if "\n" in v or "\r" in v:  # URLs don't have newlines
            logger.warning("Rejecting pdf_url: contains newlines")
            return None
        try:
            parsed = urlparse(v)
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"Rejecting pdf_url: invalid scheme '{parsed.scheme}'")
                return None
            if not parsed.netloc:
                logger.warning("Rejecting pdf_url: no netloc")
                return None
            return v
        except Exception:
            logger.warning(f"Rejecting pdf_url: failed to parse")
            return None


class BatchClassificationResponse(BaseModel):
    """Response containing classifications for multiple pages."""

    items: list[ClassificationItem] = Field(
        description="List of classification results, one for each input item"
    )


# =============================================================================
# Prompts
# =============================================================================

CLASSIFICATION_SYSTEM_PROMPT = """You are an academic content classifier. Analyze scraped web content from academic publisher pages and classify each one.

Classifications:
- full_text: The content contains the complete article body with sections like Introduction, Methods, Results, Discussion, Conclusion. Has substantial academic text across multiple sections with detailed content.
- abstract_with_pdf: The page shows only an abstract/summary with a link to download the full PDF. Look for "Download PDF", "Full Text (PDF)", "Get PDF", "View PDF", or .pdf links in the links list. The content is SHORT (just abstract + metadata).
- paywall: Shows a paywall, login requirement, subscription notice. Indicators: "Subscribe", "Purchase", "Sign in", "Institutional access", "Access denied", "You do not have access".

IMPORTANT: When classifying as abstract_with_pdf, you MUST extract the actual PDF download URL from the links provided. Choose the most direct PDF link (prefer links containing ".pdf" or "pdf/"). If no PDF URL is found but it looks like an abstract page, still classify as abstract_with_pdf with pdf_url=null.

Return one ClassificationItem for each input DOI."""

BATCH_PROMPT_TEMPLATE = """Classify each of these {count} academic article pages.

{items_text}

Return a classification for each DOI listed above."""


# =============================================================================
# Quick Heuristics
# =============================================================================


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


# =============================================================================
# Batch Classification Function
# =============================================================================


async def classify_scraped_content_batch(
    items: list[tuple[str, str, str, list[str]]],  # (doi, url, markdown, links)
) -> dict[str, ClassificationItem]:
    """Classify multiple scraped pages in a single LLM call.

    Args:
        items: List of (doi, url, markdown, links) tuples to classify.
               Max 20 items per call for cost efficiency.

    Returns:
        Dict mapping DOI to ClassificationItem
    """
    if not items:
        return {}

    if len(items) > 20:
        logger.warning(f"Batch size {len(items)} exceeds recommended max of 20")

    # Quick heuristics first - separate obvious cases
    needs_llm: list[tuple[str, str, str, list[str]]] = []
    results: dict[str, ClassificationItem] = {}

    for doi, url, markdown, links in items:
        if _quick_paywall_check(markdown):
            logger.debug(f"Quick paywall detection for {doi}")
            results[doi] = ClassificationItem(
                doi=doi,
                classification="paywall",
                confidence=0.95,
                pdf_url=None,
                reasoning="Content contains clear paywall/access restriction indicators",
            )
        elif len(markdown) > 20000 and _has_article_structure(markdown):
            logger.debug(f"Quick full_text detection for {doi}: {len(markdown)} chars with structure")
            results[doi] = ClassificationItem(
                doi=doi,
                classification="full_text",
                confidence=0.9,
                pdf_url=None,
                reasoning="Long content (>20k chars) with typical article section structure",
            )
        else:
            needs_llm.append((doi, url, markdown, links))

    if not needs_llm:
        logger.info(f"All {len(items)} items classified via heuristics")
        return results

    logger.info(f"Classifying {len(needs_llm)} items via Haiku (heuristics handled {len(results)})")

    # Build batch prompt
    items_text = _build_items_text(needs_llm)

    prompt = BATCH_PROMPT_TEMPLATE.format(count=len(needs_llm), items_text=items_text)

    try:
        response: BatchClassificationResponse = await get_structured_output(
            output_schema=BatchClassificationResponse,
            user_prompt=prompt,
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            tier=ModelTier.HAIKU,
            max_tokens=4096,
        )

        # Add LLM results to our results dict
        for item in response.items:
            results[item.doi] = item
            logger.debug(
                f"LLM classified {item.doi}: {item.classification} "
                f"(conf={item.confidence:.2f}, pdf_url={item.pdf_url is not None})"
            )

        # Check for any DOIs that weren't in the response
        expected_dois = {doi for doi, _, _, _ in needs_llm}
        returned_dois = {item.doi for item in response.items}
        missing_dois = expected_dois - returned_dois

        if missing_dois:
            logger.warning(f"LLM response missing {len(missing_dois)} DOIs: {missing_dois}")
            for doi in missing_dois:
                results[doi] = ClassificationItem(
                    doi=doi,
                    classification="full_text",
                    confidence=0.5,
                    pdf_url=None,
                    reasoning="Missing from LLM response, defaulting to full_text",
                )

    except Exception as e:
        logger.error(f"Batch classification failed: {type(e).__name__}: {e}")
        # Default all LLM-needing items to full_text on error
        for doi, url, markdown, links in needs_llm:
            results[doi] = ClassificationItem(
                doi=doi,
                classification="full_text",
                confidence=0.5,
                pdf_url=None,
                reasoning=f"Classification error, defaulting to full_text: {e}",
            )

    return results


def _build_items_text(items: list[tuple[str, str, str, list[str]]]) -> str:
    """Build the items section of the batch prompt."""
    parts = []

    for i, (doi, url, markdown, links) in enumerate(items, 1):
        # Truncate markdown to 15000 chars
        content_preview = markdown[:15000] if len(markdown) > 15000 else markdown

        # Limit to 20 links
        links_subset = links[:20] if links else []
        links_text = "\n".join(f"  - {link}" for link in links_subset) if links_subset else "  (no links found)"

        part = f"""---
Item {i}:
DOI: {doi}
URL: {url}
Content length: {len(markdown)} chars

Content (first 15000 chars):
{content_preview}

Links found on page:
{links_text}
---"""
        parts.append(part)

    return "\n\n".join(parts)


# =============================================================================
# Single-Item Convenience Function
# =============================================================================


async def classify_scraped_content(
    doi: str,
    url: str,
    markdown: str,
    links: list[str],
) -> ClassificationItem:
    """Classify a single scraped page.

    Convenience wrapper around classify_scraped_content_batch for single items.

    Args:
        doi: DOI of the paper
        url: Original URL that was scraped
        markdown: Scraped markdown content
        links: List of links found on the page

    Returns:
        ClassificationItem with type and optional PDF URL
    """
    results = await classify_scraped_content_batch([(doi, url, markdown, links)])
    return results.get(
        doi,
        ClassificationItem(
            doi=doi,
            classification="full_text",
            confidence=0.5,
            pdf_url=None,
            reasoning="Fallback: single item classification returned no result",
        ),
    )
