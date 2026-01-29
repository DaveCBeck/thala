---
module: core/scraping
date: 2026-01-14
problem_type: error_handling
component: scraping_service
symptoms:
  - "JSONDecodeError when LLM response contains text before/after JSON"
  - "Playwright triggered on PDF URLs causing download dialogs"
  - "HTTP 403/418/429 errors not triggering fallback"
  - "PDF download URLs treated as web pages"
root_cause: incomplete_error_handling
resolution_type: code_fix
severity: high
tags: [scraping, pdf, json-parsing, anti-bot, playwright, fallback, marker, http-errors]
---

# Scraping and PDF Processing Robustness Fixes

## Problem

Three interconnected issues reduced reliability of web scraping and PDF processing:

1. **JSON Extraction Failures**: LLM responses with preamble/postamble caused `JSONDecodeError`
2. **Inappropriate Scraping**: Web scraper triggered on PDF URLs, causing browser download dialogs
3. **Missing Fallbacks**: HTTP 4xx errors (403, 418, 429) didn't trigger Playwright fallback
4. **PDF Downloads Unhandled**: URLs that triggered PDF downloads failed silently

### Symptoms

```python
# Symptom 1: JSON parsing failure
JSONDecodeError: "Here's the analysis:\n```json\n{...}\n```\nExplanation..."

# Symptom 2: Playwright error on PDF URL
TimeoutError: Navigation timeout exceeded (PDF download dialog opened)

# Symptom 3: Anti-bot blocking
HTTP Error 403 Forbidden  # No fallback attempted

# Symptom 4: PDF download detection
Playwright rendered empty page (PDF download triggered instead of content)
```

## Root Cause

### 1. Naive JSON Extraction
The original parser used fixed-position slicing that assumed exact markdown format:

```python
# ❌ BEFORE: Brittle markdown parsing
def extract_json_from_llm_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])  # Assumes exact 2-line format
    return json.loads(content)  # Fails on extra text
```

### 2. No PDF URL Differentiation
Scraping fallback chain didn't distinguish PDF URLs from web pages:

```python
# ❌ BEFORE: PDF URLs fell through to web scraping
if is_pdf_url(resolved_url):
    result = await _handle_pdf_url(...)
    if not result:
        # Fall through to web scraping (WRONG for PDFs)
        pass

# Step 3: Web Scraping (called even for PDFs)
scrape_result = await scraper.scrape(resolved_url)
```

### 3. Narrow Error Pattern Matching
Only one error pattern triggered Playwright fallback:

```python
# ❌ BEFORE: Only checked for "not a valid PDF"
except MarkerProcessingError as e:
    if "not a valid PDF" in str(e):
        # Try Playwright
    else:
        raise  # HTTP 403/418/429 would raise here
```

### 4. No Download Interception
Playwright wasn't configured to detect PDF downloads:

```python
# ❌ BEFORE: Downloads not intercepted
context = await browser.new_context(
    # accept_downloads not set (defaults to False)
)
# Navigation would timeout on PDF download URLs
```

## Solution

### Fix 1: Robust JSON Extraction with Brace Matching

Three-tier parsing: direct → brace-match → error:

```python
def extract_json_from_llm_response(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown and extra text."""
    content = content.strip()

    # Remove markdown code blocks (robust matching)
    if content.startswith("```"):
        lines = content.split("\n")
        # Find closing ``` searching backwards
        end_idx = -1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        if end_idx > 0:
            content = "\n".join(lines[1:end_idx])
        else:
            content = "\n".join(lines[1:-1])
        content = content.strip()

    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find JSON by matching braces (handles extra text)
    start_idx = content.find("{")
    if start_idx == -1:
        raise json.JSONDecodeError("No JSON object found", content, 0)

    # String-aware brace counting
    depth = 0
    in_string = False
    escape_next = False
    end_idx = start_idx

    for i, char in enumerate(content[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if depth != 0:
        raise json.JSONDecodeError("Unbalanced braces", content, len(content))

    return json.loads(content[start_idx:end_idx + 1])
```

### Fix 2: Skip Web Scraping for PDF URLs

PDF URLs bypass scraper and jump directly to retrieve-academic on failure:

```python
# ✅ AFTER: PDF URLs skip web scraping
is_pdf = is_pdf_url(resolved_url)
if is_pdf:
    fallback_chain.append("pdf_direct")
    result = await _handle_pdf_url(resolved_url, doi_info, opts, fallback_chain)
    if result:
        return result

    # PDF failed - skip web scraping (won't work), try retrieve-academic
    logger.warning("PDF download failed, skipping web scraping for PDF URL")
    if opts.allow_retrieve_academic and doi_info:
        fallback_chain.append("retrieve_academic")
        result = await try_retrieve_academic(doi_info.doi, opts, fallback_chain)
        if result:
            return result

    raise Exception(f"Failed to retrieve PDF (attempted: {' -> '.join(fallback_chain)})")

# Step 3: Web Scraping (only for non-PDF URLs)
```

### Fix 3: Extended Fallback Pattern Matching

Playwright fallback for HTTP errors, timeouts, and non-PDF responses:

```python
async def _download_pdf(url: str, timeout: float = 60.0) -> bytes:
    """Download PDF with extended fallback triggers."""
    try:
        return await _download_pdf_httpx(url, timeout)
    except MarkerProcessingError as e:
        error_str = str(e)
        # ✅ AFTER: Broader error matching
        if any(pattern in error_str for pattern in [
            "not a valid PDF",    # HTML redirect/login page
            "HTTP error",         # 403, 418, 429 anti-bot
            "Timeout",           # Slow sites
        ]):
            logger.debug(f"httpx failed ({e}), trying Playwright fallback")
        else:
            raise

    # Fallback to Playwright
    return await _download_pdf_playwright(url, timeout)
```

### Fix 4: PDF Download Detection in Playwright

Custom exception and download interception:

```python
class PDFDownloadDetected(Exception):
    """Raised when PDF download detected instead of web page."""
    def __init__(self, content: bytes, url: str):
        self.content = content
        self.url = url
        super().__init__(f"PDF download detected ({len(content)} bytes)")


async def scrape(self, url: str) -> str:
    """Scrape URL, detecting PDF downloads."""
    context = await browser.new_context(
        accept_downloads=True,  # Enable download detection
    )
    page = await context.new_page()
    download_content: bytes | None = None

    async def handle_download(download):
        nonlocal download_content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            await download.save_as(tmp.name)
            download_content = Path(tmp.name).read_bytes()

    page.on("download", handle_download)

    try:
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(0.5)  # Allow download to start

        if download_content:
            raise PDFDownloadDetected(download_content, url)

        return await page.content()

    except PDFDownloadDetected:
        raise
```

**Service layer integration:**

```python
async def _scrape_playwright(self, url: str) -> ScrapeResult:
    """Scrape with PDF download handling."""
    try:
        markdown = await scraper.scrape(url)
    except PDFDownloadDetected as e:
        # Convert captured PDF to markdown via Marker
        from .pdf import process_pdf_bytes
        markdown = await process_pdf_bytes(e.content)
        return ScrapeResult(url=url, markdown=markdown, provider="playwright-pdf")

    return ScrapeResult(url=url, markdown=markdown, provider="playwright")
```

## Files Modified

**JSON extraction:**
- `workflows/research/web_research/utils/json_utils.py` - Three-tier JSON parser

**PDF URL handling:**
- `core/scraping/unified.py` - Skip web scraping for PDF URLs
- `core/scraping/pdf/processor.py` - Extended fallback patterns

**Playwright PDF detection:**
- `core/scraping/playwright_scraper.py` - Download interception, PDFDownloadDetected
- `core/scraping/service.py` - Handle PDFDownloadDetected in service layer

## Prevention

### JSON Extraction
- Always use `extract_json_from_llm_response()` for LLM outputs
- Consider structured output schemas for critical extractions

### URL Type Detection
- Check URL type before choosing processing method
- PDF URLs should never reach web scraper

### Anti-Bot Handling
- Match multiple error patterns for fallback decisions
- Include status code ranges (4xx), timeouts, content validation failures

### Download Detection
- Configure `accept_downloads=True` for Playwright contexts
- Handle `PDFDownloadDetected` in all scraping code paths

## Complete Error Flow

```
URL Input
    │
    ├─[PDF URL?]─YES──▶ PDF Download (httpx)
    │                     │
    │                     ├─403/418/429?──▶ Playwright Download
    │                     │                    │
    │                     │                    └─▶ Convert via Marker
    │                     │
    │                     └─Failed?──▶ retrieve-academic (skip scraping)
    │
    └─NO──▶ Web Scraping
               │
               ├─Firecrawl Local
               │    └─Blocked?──▶ Firecrawl Cloud Stealth
               │                     └─Blocked?──▶ Playwright
               │                                      │
               └───────────────────────────────────────┘
                                                       │
                                              [PDF Download Detected?]
                                                       │
                                                 YES──▶ Convert via Marker
                                                       │
                                                  NO──▶ Return HTML content
```

## Related Patterns

- [Unified Scraping Service with Fallback Chain](../../patterns/data-pipeline/unified-scraping-service-fallback-chain.md) - Core fallback architecture
- [Unified Content Retrieval Pipeline](../../patterns/data-pipeline/unified-content-retrieval-pipeline.md) - get_url() pipeline
- [GPU-Accelerated Document Processing](../../patterns/data-pipeline/gpu-accelerated-document-processing.md) - Marker integration

## Related Solutions

- [Paper Acquisition Robustness](../api-integration-issues/paper-acquisition-robustness.md) - Multi-source paper retrieval
- [Large Document Processing](../api-integration-issues/large-document-processing.md) - Chunking for large PDFs

## References

- [Playwright Downloads](https://playwright.dev/python/docs/downloads)
- [httpx Error Handling](https://www.python-httpx.org/exceptions/)
