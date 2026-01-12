"""Content classification for scraped HTML pages.

Uses Haiku for fast, cheap batch classification of whether scraped content is:
- full_text: Complete article ready for processing
- abstract_with_pdf: Abstract page with PDF download link
- paywall: Access restricted, needs fallback to retrieve-academic
"""

import logging
import re

from workflows.shared.llm_utils import ModelTier, get_structured_output

from .prompts import BATCH_PROMPT_TEMPLATE, CLASSIFICATION_SYSTEM_PROMPT
from .types import BatchClassificationResponse, ClassificationItem

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
