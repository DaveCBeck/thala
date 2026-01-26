# core/scraping - Unified URL Content Retrieval

## Quick Start

```python
from core.scraping import get_url, GetUrlResult, GetUrlOptions

# Simple usage - handles URLs, DOIs, PDFs automatically
result = await get_url("https://example.com")
result = await get_url("10.1038/nature12373")  # bare DOI
result = await get_url("https://arxiv.org/pdf/2301.00001.pdf")  # PDF

# With options
result = await get_url(url, GetUrlOptions(pdf_quality="quality"))

# Low-level scraping service (for direct control)
from core.scraping import get_scraper_service
service = get_scraper_service()
result = await service.scrape("https://example.com")

# PDF processing by MD5 hash (via retrieve-academic)
from core.scraping import download_pdf_by_md5, process_pdf_by_md5
path = await download_pdf_by_md5(md5_hash, output_path)
markdown = await process_pdf_by_md5(md5_hash)
```

## Flow

```
get_url(url)
    │
    ├─► DOI Detection (bare DOI, doi.org URL, publisher URLs)
    │   └─► OpenAlex lookup → OA URL resolution
    │
    ├─► PDF Detection (.pdf extension)
    │   └─► Download (httpx → Playwright fallback) → Marker → markdown
    │
    ├─► Web Scraping (3-tier cascade)
    │   └─► Local Firecrawl → Cloud Stealth → Playwright
    │
    ├─► Content Classification (DeepSeek V3 via get_structured_output)
    │   ├─► full_text → return markdown
    │   ├─► abstract_with_pdf → extract PDF URL → Marker
    │   ├─► paywall → title search (see below)
    │   └─► non_academic → return markdown
    │
    ├─► DOI Title Search (if paywall + no DOI)
    │   └─► Extract title/authors from page → OpenAlex search → DOI
    │
    └─► retrieve-academic Fallback (if DOI known/found)
        └─► Submit DOI → poll → download PDF → Marker → markdown
```

## Module Structure

```
core/scraping/
├── __init__.py             # Public exports
├── unified.py              # get_url() entry point
├── types.py                # GetUrlResult, GetUrlOptions, ContentSource, DoiInfo
├── config.py               # FirecrawlConfig
├── errors.py               # ScrapingError, SiteBlockedError, etc.
├── service.py              # ScraperService (3-tier cascade)
├── firecrawl_clients.py    # FirecrawlClients manager
├── playwright_scraper.py   # PlaywrightScraper fallback
├── doi/
│   ├── __init__.py
│   ├── detector.py         # DOI regex, publisher URL patterns
│   └── resolver.py         # OpenAlex API
├── pdf/
│   ├── __init__.py
│   ├── detector.py         # is_pdf_url()
│   └── processor.py        # Marker client + Playwright PDF download
├── classification/
│   ├── __init__.py
│   ├── classifier.py       # Content classifier (DeepSeek V3)
│   ├── prompts.py          # Classification prompts
│   └── types.py            # ClassificationResult
└── fallback/
    ├── __init__.py
    └── academic.py         # retrieve-academic integration
```

## Result Types

```python
class GetUrlResult:
    url: str                        # Original input
    resolved_url: str | None        # After DOI/redirect resolution
    content: str                    # Markdown output
    content_type: str               # Always "markdown" for now
    source: ContentSource           # scraped|pdf_direct|pdf_extracted|retrieve_academic
    provider: str                   # firecrawl-local|marker|retrieve-academic|etc
    doi: str | None                 # Detected/resolved DOI
    classification: ContentClassification | None  # full_text|paywall|etc
    links: list[str]                # Extracted links from page
    fallback_chain: list[str]       # Debug: attempted sources
```

## DOI Detection

Handles:
- Bare DOI: `10.1234/example`
- DOI URL: `https://doi.org/10.1234/example`
- Publisher URLs: Springer, Wiley, PNAS, Oxford Academic, T&F, SAGE, ACS, RSC, IOP, Cambridge

## Environment Variables

### Firecrawl Configuration
- `FIRECRAWL_LOCAL_URL`: URL of self-hosted Firecrawl (e.g., `http://localhost:3002`)
- `FIRECRAWL_API_KEY`: API key for cloud Firecrawl (required for stealth fallback)
- `FIRECRAWL_TIMEOUT`: Request timeout in seconds (default: 45)
- `FIRECRAWL_SKIP_LOCAL`: Set to `true` to skip local and use cloud only

### Marker Configuration
- `MARKER_BASE_URL`: Marker service URL (default: `http://localhost:8001`)
- `MARKER_INPUT_DIR`: Directory for PDF input files (default: `/data/input`)
- `MARKER_POLL_INTERVAL`: Job polling interval in seconds (default: 2.0)

## Dependencies

- **Marker service**: PDF → markdown conversion
- **retrieve-academic** (optional): Paywalled content fallback
- **OpenAlex API**: DOI resolution (no auth required)
- **Firecrawl**: Web scraping (local + cloud)
