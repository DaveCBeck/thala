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
    ├─► Content Classification (Haiku via Anthropic SDK)
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
├── unified.py          # get_url() entry point
├── types.py            # GetUrlResult, GetUrlOptions, enums
├── doi/
│   ├── detector.py     # DOI regex, publisher URL patterns
│   └── resolver.py     # OpenAlex API
├── pdf/
│   ├── detector.py     # is_pdf_url()
│   └── processor.py    # Marker client + Playwright PDF download
├── classification/
│   ├── classifier.py   # Anthropic SDK classifier (no LangChain)
│   └── prompts.py      # Classification prompts
├── fallback/
│   └── academic.py     # retrieve-academic integration
└── service.py          # ScraperService (existing 3-tier cascade)
```

## Result Types

```python
class GetUrlResult:
    url: str                        # Original input
    resolved_url: str               # After DOI/redirect resolution
    content: str                    # Markdown output
    source: ContentSource           # scraped|pdf_direct|pdf_extracted|retrieve_academic
    provider: str                   # firecrawl-local|marker|retrieve-academic|etc
    doi: str | None                 # Detected/resolved DOI
    classification: ContentClassification | None  # full_text|paywall|etc
    fallback_chain: list[str]       # Debug: attempted sources
```

## DOI Detection

Handles:
- Bare DOI: `10.1234/example`
- DOI URL: `https://doi.org/10.1234/example`
- Publisher URLs: Springer, Wiley, PNAS, Nature, T&F, SAGE, ACS, RSC, IOP, Cambridge

## Dependencies

- **Marker service** (localhost:8001): PDF → markdown conversion
- **retrieve-academic** (optional): Paywalled content fallback
- **OpenAlex API**: DOI resolution (no auth required)
- **Firecrawl**: Web scraping (local + cloud)
