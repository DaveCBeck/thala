---
module: paper_processor
date: 2026-01-05
problem_type: content_type_detection
component: classification
symptoms:
  - "Scraped HTML pages incorrectly processed as full text"
  - "Abstract pages missing PDF download links"
  - "Paywall pages causing processing failures"
  - "No differentiation between content types from OA URLs"
root_cause: missing_content_classification
resolution_type: llm_classification
severity: medium
tags: [classification, html, pdf, paywall, haiku, heuristics, structured-output]
---

# HTML Content Classification

## Problem

Scraped HTML content from OA URLs was processed uniformly without identifying content type:

- **Full text pages**: Should be processed directly
- **Abstract pages**: Have PDF download links that should be extracted
- **Paywall pages**: Need fallback to retrieve-academic

Without classification, abstract pages were processed as if they contained full text (missing the actual paper content), and paywall pages caused processing failures.

## Solution

**Add LLM-based content classification with heuristic pre-filtering.**

Classify scraped pages into three categories:
1. `full_text` - Complete article ready for processing
2. `abstract_with_pdf` - Abstract page with PDF link to extract
3. `paywall` - Access restricted, needs fallback

### Architecture

```
Scraped Content
      │
      ▼
┌─────────────────────────────┐
│  Quick Heuristics (free)    │
│  - Paywall indicators       │
│  - Article structure check  │
└─────────────┬───────────────┘
              │
    ┌─────────┴─────────┐
    │ Obvious?          │
    │                   │
Yes ▼                   ▼ No
┌───────────────┐  ┌─────────────────┐
│ Return result │  │ Haiku batch LLM │
└───────────────┘  │ classification  │
                   └─────────────────┘
```

## Implementation

### Pydantic Schemas

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor/classification.py

from typing import Literal, Optional
from pydantic import BaseModel, Field


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
        description="URL to PDF if classification is abstract_with_pdf",
    )
    reasoning: str = Field(
        description="Brief explanation for this classification",
    )


class BatchClassificationResponse(BaseModel):
    """Response containing classifications for multiple pages."""

    items: list[ClassificationItem] = Field(
        description="List of classification results"
    )
```

### Quick Heuristics

Fast, free checks before using LLM:

```python
def _quick_paywall_check(markdown: str) -> bool:
    """Quick heuristic check for obvious paywalls."""
    markdown_lower = markdown.lower()
    paywall_indicators = [
        "sign in to access",
        "subscribe to read",
        "purchase this article",
        "institutional access required",
        "access denied",
        "you do not have access",
        "login required",
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
    ]
    matches = sum(1 for p in section_patterns if re.search(p, markdown, re.IGNORECASE))
    return matches >= 3  # At least 3 typical sections
```

### Batch Classification

```python
async def classify_scraped_content_batch(
    items: list[tuple[str, str, str, list[str]]],  # (doi, url, markdown, links)
) -> dict[str, ClassificationItem]:
    """Classify multiple scraped pages in a single LLM call."""
    if not items:
        return {}

    # Quick heuristics first
    needs_llm: list[tuple[str, str, str, list[str]]] = []
    results: dict[str, ClassificationItem] = {}

    for doi, url, markdown, links in items:
        if _quick_paywall_check(markdown):
            results[doi] = ClassificationItem(
                doi=doi,
                classification="paywall",
                confidence=0.95,
                pdf_url=None,
                reasoning="Clear paywall/access restriction indicators",
            )
        elif len(markdown) > 20000 and _has_article_structure(markdown):
            results[doi] = ClassificationItem(
                doi=doi,
                classification="full_text",
                confidence=0.9,
                pdf_url=None,
                reasoning="Long content with article section structure",
            )
        else:
            needs_llm.append((doi, url, markdown, links))

    if not needs_llm:
        return results

    logger.info(f"Classifying {len(needs_llm)} items via Haiku")

    # Build batch prompt
    items_text = _build_items_text(needs_llm)
    prompt = BATCH_PROMPT_TEMPLATE.format(count=len(needs_llm), items_text=items_text)

    # Use Haiku with structured output
    llm = get_llm(tier=ModelTier.HAIKU, max_tokens=4096)
    structured_llm = llm.with_structured_output(BatchClassificationResponse)

    response: BatchClassificationResponse = await structured_llm.ainvoke([
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    for item in response.items:
        results[item.doi] = item

    return results
```

### Classification Prompt

```python
CLASSIFICATION_SYSTEM_PROMPT = """You are an academic content classifier.

Classifications:
- full_text: Complete article with Introduction, Methods, Results, Discussion, Conclusion.
- abstract_with_pdf: Abstract/summary with PDF download link. Look for "Download PDF", ".pdf" links.
- paywall: Login requirement, subscription notice, access denied.

IMPORTANT: For abstract_with_pdf, extract the actual PDF URL from the links provided.
"""
```

### Integration with Acquisition Pipeline

```python
# In acquisition.py

async def process_oa_result(doi: str, source: str, is_markdown: bool, links: list[str]):
    """Process OA acquisition result with classification."""
    if not is_markdown:
        # PDF file - process directly
        return await process_pdf(source)

    # HTML content - classify first
    classification = await classify_scraped_content(doi, url, source, links)

    if classification.classification == "paywall":
        # Need fallback to retrieve-academic
        return await fallback_to_retrieve_academic(doi)

    elif classification.classification == "abstract_with_pdf":
        if classification.pdf_url:
            # Download and process the PDF
            pdf_path = await download_pdf(classification.pdf_url)
            return await process_pdf(pdf_path)
        else:
            # No PDF URL found - fallback
            return await fallback_to_retrieve_academic(doi)

    else:  # full_text
        # Process the markdown directly
        return await process_markdown(source)
```

## Classification Criteria

| Classification | Indicators | Confidence |
|----------------|------------|------------|
| `paywall` | Access restriction text | 0.95 (heuristic), 0.8+ (LLM) |
| `full_text` | >20k chars + 3+ sections | 0.9 (heuristic), 0.8+ (LLM) |
| `abstract_with_pdf` | Short + PDF links | 0.8+ (LLM only) |

## Cost Efficiency

- **Heuristics first**: Free, handles ~50% of cases
- **Batch LLM calls**: Haiku is cheap, batch reduces overhead
- **Max 20 items per batch**: Balance cost vs context window

## Files Modified

- `workflows/research/subgraphs/academic_lit_review/paper_processor/classification.py` - New module
- `workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py` - Integration

## Prevention

When processing mixed content types:

1. **Always classify first**: Don't assume content type from URL
2. **Use heuristics for obvious cases**: Save LLM costs
3. **Handle missing PDF URLs**: Fallback to alternative acquisition
4. **Batch classifications**: More efficient than individual calls

## Related Solutions

- [Multi-Strategy Paper Acquisition](../api-integration-issues/multi-strategy-paper-acquisition.md) - OA URL handling
- [Batch API JSON/Structured Output](../llm-issues/batch-api-json-structured-output.md) - Structured output patterns

## References

- [Pydantic Structured Output](https://docs.pydantic.dev/)
