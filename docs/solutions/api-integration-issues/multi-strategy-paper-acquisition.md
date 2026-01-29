---
module: paper_processor
date: 2026-01-04
problem_type: api_integration_issue
component: acquisition
symptoms:
  - "Slow paper acquisition relying solely on retrieve-academic"
  - "Papers with OA URLs not using faster direct download"
  - "HTML papers not being processed (only PDF supported)"
root_cause: single_acquisition_strategy
resolution_type: multi_strategy_fallback
severity: medium
tags: [acquisition, openalex, open-access, firecrawl, fallback, pdf, html]
---

# Multi-Strategy Paper Acquisition

## Problem

Paper acquisition was slow and limited:
- Only used retrieve-academic service (VPN-based retrieval)
- Ignored OpenAlex's open access URLs when available
- HTML papers (not PDF) couldn't be processed
- No differentiation between PDF and HTML content types

## Solution

**Implement multi-strategy acquisition with OA URL priority:**

1. **Try OA URL first** (if available from OpenAlex)
2. **Handle both PDF and HTML** content types
3. **Fall back to retrieve-academic** if OA fails

### Acquisition Strategy Order

```
┌─────────────────────────────────────────────────────────────────┐
│ Paper has oa_url?                                               │
│   ├── Yes → Is URL a PDF? (.pdf extension)                      │
│   │         ├── Yes → Direct HTTP download                      │
│   │         │         └── Success? Use PDF path                 │
│   │         │         └── Fail? → Fall to retrieve-academic     │
│   │         └── No → Scrape via firecrawl                       │
│   │                  └── Success? Use markdown content          │
│   │                  └── Fail? → Fall to retrieve-academic      │
│   └── No → Use retrieve-academic directly                       │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### URL Type Detection

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py

def _is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    # Check URL path for .pdf extension (ignore query params)
    clean_url = url.lower().split("?")[0].split("#")[0].rstrip("/")
    return clean_url.endswith(".pdf")
```

### OA Download Function

```python
OA_DOWNLOAD_TIMEOUT = 60.0

async def try_oa_download(
    oa_url: str,
    local_path: Path,
    doi: str,
) -> tuple[Optional[str], bool]:
    """Try to download paper from Open Access URL.

    Handles both PDF URLs (direct download) and HTML URLs (firecrawl scrape).

    Returns:
        Tuple of (source, is_markdown):
        - For PDF: (local_path_str, False) on success
        - For HTML: (markdown_content, True) on success
        - (None, False) on failure
    """
    try:
        if _is_pdf_url(oa_url):
            # Direct PDF download
            logger.info(f"[OA] Downloading PDF for {doi}: {oa_url}")
            async with httpx.AsyncClient(timeout=OA_DOWNLOAD_TIMEOUT) as client:
                response = await client.get(oa_url, follow_redirects=True)
                response.raise_for_status()

                # Verify it's actually a PDF
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not response.content[:4] == b"%PDF":
                    logger.warning(f"[OA] URL returned non-PDF content: {content_type}")
                    return None, False

                with open(local_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"[OA] Downloaded PDF: {len(response.content) / 1024:.1f} KB")
                return str(local_path), False

        else:
            # HTML page - scrape with firecrawl
            logger.info(f"[OA] Scraping HTML page for {doi}: {oa_url}")
            response = await scrape_url.ainvoke({"url": oa_url})
            markdown = response.get("markdown", "")

            if not markdown or len(markdown) < 500:
                logger.warning(f"[OA] Scraped content too short: {len(markdown)} chars")
                return None, False

            logger.info(f"[OA] Scraped HTML: {len(markdown)} chars")
            return markdown, True

    except httpx.HTTPStatusError as e:
        logger.warning(f"[OA] HTTP error for {doi}: {e.response.status_code}")
        return None, False
    except Exception as e:
        logger.warning(f"[OA] Failed for {doi}: {type(e).__name__}: {e}")
        return None, False
```

### Integration with Acquisition Pipeline

```python
async def try_acquire_single(
    paper: PaperMetadata, index: int
) -> tuple[str, Optional[str], Optional[str], bool]:
    """Try OA first, then submit to retrieve-academic if needed.

    Returns:
        (doi, job_id_or_none, local_path_or_content, is_markdown)
        - If OA succeeded: (doi, None, source, is_markdown)
        - If needs retrieve: (doi, job_id, local_path, False)
    """
    async with semaphore:
        if index > 0:
            await asyncio.sleep(ACQUISITION_DELAY)

        doi = paper.get("doi")
        oa_url = paper.get("oa_url")

        safe_doi = doi.replace("/", "_").replace(":", "_")
        local_path = output_dir / f"{safe_doi}.pdf"

        # Try OA download first if URL available
        if oa_url:
            source, is_markdown = await try_oa_download(oa_url, local_path, doi)
            if source:
                oa_acquired_count += 1
                return doi, None, source, is_markdown

        # Fall back to retrieve-academic
        authors = [a.get("name") for a in paper.get("authors", [])[:5]]
        job = await client.retrieve(
            doi=doi,
            title=paper.get("title", "Unknown"),
            authors=authors,
        )

        return doi, job.job_id, str(local_path), False


# Process results
for result in submit_results:
    doi, job_id, source, is_markdown = result
    if job_id is None:
        # OA success - push directly to processing queue
        await processing_queue.put((doi, source, paper, is_markdown))
    else:
        # Needs retrieve-academic polling
        valid_jobs.append(result)
```

### Processing Pipeline Update

The processing pipeline now accepts an `is_markdown` flag:

```python
# Document processing now handles both PDF paths and markdown content
await processing_queue.put((doi, source, paper, is_markdown))

# In consumer:
doi, source, paper, is_markdown = item
if is_markdown:
    # Source is markdown content string
    result = await process_markdown_content(source, paper)
else:
    # Source is local PDF path
    result = await process_pdf_document(source, paper)
```

## Content Type Handling

| Content Type | Detection | Processing |
|--------------|-----------|------------|
| PDF (direct URL) | `.pdf` extension | Download → Marker → Text |
| PDF (content-type) | `application/pdf` header | Download → Marker → Text |
| HTML page | No `.pdf` extension | Firecrawl → Markdown |

## Validation

PDF downloads are validated before saving:
```python
# Check content-type header
content_type = response.headers.get("content-type", "").lower()
if "pdf" not in content_type:
    # Check magic bytes
    if response.content[:4] != b"%PDF":
        return None, False  # Not a PDF
```

## Files Modified

- `workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py` - Multi-strategy acquisition
- `workflows/research/subgraphs/academic_lit_review/paper_processor/document_processing.py` - Handle markdown content

## Related Fixes in Same Commit

- Fixed batch API custom_id validation (sanitize DOI slashes/dots)
- Fixed ES term queries to use `.keyword` suffix
- Increased Marker task timeouts (3h soft, 4h hard) for large PDFs
- Disabled `COMPILE_ALL` in Marker (causes failures with variable page sizes)

## Prevention

When adding acquisition strategies:
1. **Prioritize free/fast sources**: OA URLs are faster than VPN-based retrieval
2. **Handle content type variance**: Same URL might return PDF or HTML
3. **Validate content before processing**: Check magic bytes, not just URL extension
4. **Track acquisition source**: Log which strategy succeeded for metrics

## Related Solutions

- [Paper Acquisition Robustness](./paper-acquisition-robustness.md) - ES cache and retry patterns
- [Search Reliability: Multi-Query Fallback](./search-reliability-multi-query-fallback.md) - Similar fallback strategy

## References

- [OpenAlex API - Open Access](https://docs.openalex.org/api-entities/works/work-object#open_access)
- [Firecrawl Scraping](https://docs.firecrawl.dev/)
