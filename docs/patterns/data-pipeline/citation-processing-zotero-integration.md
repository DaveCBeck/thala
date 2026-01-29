---
name: citation-processing-zotero-integration
title: "Citation Processing with Zotero Integration"
date: 2025-12-19
category: data-pipeline
applicability:
  - "When enriching web URLs with bibliographic metadata"
  - "When integrating with Zotero Translation Server for citation extraction"
  - "When combining automated extraction with LLM gap-filling"
components: [zotero, translation_server, llm, pydantic, asyncio]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [zotero, citation, metadata, translation-server, llm-enrichment, pydantic, async]
---

# Citation Processing with Zotero Integration

## Intent

Provide reliable bibliographic metadata extraction from arbitrary URLs using Zotero Translation Server as the primary source, with LLM-based gap-filling for incomplete data and deduplication against existing Zotero collections.

## Motivation

Research workflows often need to convert raw URLs into properly formatted citations. This presents several challenges:

1. **Heterogeneous sources**: Academic papers, news articles, blog posts, and documentation all have different metadata structures
2. **Incomplete extraction**: Translation servers may extract partial metadata (e.g., title but no author)
3. **Duplicate management**: The same URL may be cited multiple times across research sessions
4. **Schema mapping**: External metadata must be transformed into Zotero's specific field format

This pattern addresses these challenges through a multi-stage pipeline that combines automated extraction with intelligent gap-filling.

## Applicability

Use this pattern when:
- You need bibliographic metadata from diverse web sources
- Zotero Translation Server is available for metadata extraction
- Some sources will have incomplete metadata requiring LLM assistance
- Deduplication against existing citations is required

Do NOT use this pattern when:
- You only need simple URL bookmarking without metadata
- All sources are from a single, well-structured API (e.g., CrossRef only)
- Real-time processing is required (translation server adds latency)

## Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     Citation Processing Pipeline                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Deduplication Check                                    │
│  ─────────────────────────────                                  │
│  Query existing Zotero items by URL                             │
│  If found → Return existing key (skip processing)               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Translation Server Extraction                          │
│  ─────────────────────────────────────                          │
│  POST /web with URL → Structured metadata                       │
│  Handles: articles, papers, news, generic web pages             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: LLM Gap-Filling                                        │
│  ───────────────────────                                        │
│  If metadata incomplete:                                        │
│    - Scrape page content                                        │
│    - Extract missing fields with structured output              │
│  Merge: Translation result + LLM extraction                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Zotero Item Creation                                   │
│  ────────────────────────────                                   │
│  Transform to Zotero schema                                     │
│  Create item via Zotero API                                     │
│  Return: Zotero item key                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Translation Server Client

Create an async client for the Zotero Translation Server:

```python
from dataclasses import dataclass
import httpx

@dataclass
class TranslationResult:
    """Result from translation server."""
    success: bool
    item_type: str | None = None
    title: str | None = None
    authors: list[dict] | None = None
    date: str | None = None
    abstract: str | None = None
    url: str | None = None
    doi: str | None = None
    raw_data: dict | None = None
    error: str | None = None


class TranslationServerClient:
    """Async client for Zotero Translation Server."""

    def __init__(self, base_url: str = "http://localhost:1969"):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def translate_url(self, url: str) -> TranslationResult:
        """Translate a URL to bibliographic metadata."""
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.base_url}/web",
                json={"url": url, "sessionid": "thala"},
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    item = data[0]
                    return TranslationResult(
                        success=True,
                        item_type=item.get("itemType"),
                        title=item.get("title"),
                        authors=item.get("creators", []),
                        date=item.get("date"),
                        abstract=item.get("abstractNote"),
                        url=item.get("url"),
                        doi=item.get("DOI"),
                        raw_data=item,
                    )

            return TranslationResult(
                success=False,
                error=f"Translation failed: {response.status_code}",
            )

        except httpx.TimeoutException:
            return TranslationResult(success=False, error="Translation timeout")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
```

### Step 2: Pydantic Model for LLM Extraction

Define a structured output schema for gap-filling:

```python
from pydantic import BaseModel, Field

class ExtractedCitationMetadata(BaseModel):
    """Metadata extracted from page content by LLM."""

    title: str | None = Field(
        default=None,
        description="The title of the article, paper, or page",
    )
    authors: list[str] | None = Field(
        default=None,
        description="List of author names in 'First Last' format",
    )
    publication_date: str | None = Field(
        default=None,
        description="Publication date in YYYY-MM-DD or YYYY format",
    )
    abstract: str | None = Field(
        default=None,
        description="Brief summary or abstract of the content",
    )
    publication_name: str | None = Field(
        default=None,
        description="Name of journal, website, or publication",
    )
    item_type: str = Field(
        default="webpage",
        description="Type: journalArticle, newspaperArticle, blogPost, webpage",
    )
```

### Step 3: Multi-Source Processing Pipeline

Implement the full pipeline with deduplication and gap-filling:

```python
import asyncio
from langchain_anthropic import ChatAnthropic

async def process_citation(
    url: str,
    translation_client: TranslationServerClient,
    store_manager: StoreManager,
    scraping_service: ScrapingService,
) -> tuple[str, str | None]:
    """
    Process a single citation URL.

    Returns:
        Tuple of (url, zotero_key or None if failed)
    """
    # Step 1: Check for existing Zotero item (deduplication)
    existing_key = await _check_existing_zotero_item(url, store_manager)
    if existing_key:
        return (url, existing_key)

    # Step 2: Get translation metadata
    translation_result = await translation_client.translate_url(url)

    # Step 3: Determine if LLM gap-filling is needed
    needs_enhancement = (
        not translation_result.success
        or not translation_result.title
        or not translation_result.authors
    )

    if needs_enhancement:
        enhanced_metadata = await _enhance_with_llm(
            url, translation_result, scraping_service
        )
    else:
        enhanced_metadata = _convert_translation_result(translation_result)

    # Step 4: Create Zotero item
    zotero_key = await _create_zotero_item(url, enhanced_metadata, store_manager)

    return (url, zotero_key)


async def _enhance_with_llm(
    url: str,
    translation_result: TranslationResult,
    scraping_service: ScrapingService,
) -> ExtractedCitationMetadata:
    """Fill metadata gaps using LLM extraction."""
    # Scrape page content
    page_content = await scraping_service.scrape(url)

    # Use fast model for extraction (cost-effective)
    llm = ChatAnthropic(model="claude-3-haiku-20240307")
    structured_llm = llm.with_structured_output(ExtractedCitationMetadata)

    prompt = f"""Extract bibliographic metadata from this content.

URL: {url}

Existing metadata (fill gaps only):
- Title: {translation_result.title or 'MISSING'}
- Authors: {translation_result.authors or 'MISSING'}
- Date: {translation_result.date or 'MISSING'}

Page content:
{page_content[:8000]}

Extract any missing fields. Keep existing values if already present."""

    extracted = await structured_llm.ainvoke(prompt)

    # Merge: Translation result takes precedence
    return ExtractedCitationMetadata(
        title=translation_result.title or extracted.title,
        authors=_merge_authors(translation_result.authors, extracted.authors),
        publication_date=translation_result.date or extracted.publication_date,
        abstract=translation_result.abstract or extracted.abstract,
        publication_name=extracted.publication_name,
        item_type=translation_result.item_type or extracted.item_type,
    )
```

### Step 4: Concurrent Processing with Rate Limiting

Process multiple citations concurrently with semaphore-based rate limiting:

```python
async def process_citations(
    urls: list[str],
    translation_client: TranslationServerClient,
    store_manager: StoreManager,
    scraping_service: ScrapingService,
    max_concurrent: int = 5,
) -> dict[str, str | None]:
    """
    Process multiple citation URLs concurrently.

    Args:
        urls: List of URLs to process
        max_concurrent: Maximum concurrent requests (respects external services)

    Returns:
        Dict mapping URL to Zotero key (or None if failed)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(url: str) -> tuple[str, str | None]:
        async with semaphore:
            try:
                return await process_citation(
                    url, translation_client, store_manager, scraping_service
                )
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                return (url, None)

    results = await asyncio.gather(
        *[process_with_semaphore(url) for url in urls],
        return_exceptions=False,
    )

    return dict(results)
```

## Consequences

### Benefits

- **Robust extraction**: Translation server handles most sources; LLM fills gaps
- **Cost-effective**: Uses fast/cheap model (Haiku) for gap-filling only when needed
- **Deduplication**: Prevents duplicate Zotero entries for repeated URLs
- **Structured output**: Pydantic ensures consistent metadata schema
- **Rate limiting**: Semaphore prevents overwhelming external services

### Trade-offs

- **Latency**: Multi-step pipeline adds processing time per URL
- **External dependency**: Requires running Translation Server instance
- **LLM costs**: Gap-filling adds cost for incomplete extractions
- **Complexity**: More moving parts than simple metadata scraping

### Alternatives

- **CrossRef-only**: Simpler but limited to DOI-based sources
- **LLM-only extraction**: More consistent but higher cost and less accurate
- **Manual entry**: Most accurate but doesn't scale

## Related Patterns

- [Deep Research Workflow Architecture](../langgraph/deep-research-workflow-architecture.md) - Uses this pattern for citation enrichment
- [Unified Scraping Service](./unified-scraping-service-fallback-chain.md) - Provides page content for LLM extraction

## Known Uses in Thala

- `core/stores/translation_server.py`: TranslationServerClient implementation
- `workflows/research/nodes/process_citations.py`: Full processing pipeline
- `workflows/research/graph.py`: Integration with research workflow

## References

- [Zotero Translation Server](https://github.com/zotero/translation-server)
- [Zotero Web API](https://www.zotero.org/support/dev/web_api/v3/basics)
- [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/)
