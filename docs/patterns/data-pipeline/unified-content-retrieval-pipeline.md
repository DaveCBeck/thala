---
name: unified-content-retrieval-pipeline
title: "Unified Content Retrieval with DOI Resolution and Academic Fallback"
date: 2026-01-13
category: data-pipeline
applicability:
  - "Acquiring academic content from diverse sources (open access, paywalled, various publishers)"
  - "URL-to-markdown conversion with intelligent content type handling"
  - "Systems requiring DOI resolution and open access URL lookup"
  - "Workflows needing graceful degradation for paywalled content"
components: [get_url, doi_detector, doi_resolver, classifier, retrieve_academic, marker, scraper_service]
complexity: high
verified_in_production: true
related_solutions: []
tags: [content-retrieval, doi-resolution, openalex, pdf-processing, paywall-detection, academic-fallback, content-classification]
---

# Unified Content Retrieval with DOI Resolution and Academic Fallback

## Intent

Provide a single entry point (`get_url()`) for all URL content retrieval that intelligently handles DOIs, PDFs, web pages, paywalls, and academic content with automatic fallback chains.

## Problem

Academic content acquisition faces multiple challenges:

1. **DOI complexity**: Bare DOIs, doi.org URLs, and publisher-specific URL patterns all need handling
2. **Access barriers**: Paywalled content requires alternative acquisition strategies
3. **Format diversity**: PDFs need extraction, web pages need scraping, abstracts need PDF link detection
4. **Publisher variation**: Different publishers use different URL structures and access patterns
5. **Content classification**: Need to distinguish full text, abstracts, paywalls, and non-academic content

## Solution

Implement a 5-step pipeline that normalizes all inputs to markdown output:

```
Step 1: DOI Detection
  └─ Detect DOI from input → Resolve via OpenAlex → Get OA URL

Step 2: PDF Detection & Processing
  └─ Direct PDF URLs → Download → Process via Marker

Step 3: Web Scraping
  └─ Firecrawl/Playwright fallback chain → Get page content

Step 4: Content Classification
  └─ Heuristics + LLM → Classify as full_text/abstract_with_pdf/paywall/non_academic

Step 5: Academic Fallback
  └─ For paywalled content with DOI → retrieve-academic service
```

### Architecture

```
get_url(url)
    │
    ├─ detect_doi(url)
    │   ├─ Bare DOI: "10.1234/example"
    │   ├─ DOI URL: "https://doi.org/10.1234/example"
    │   └─ Publisher URL: "springer.com/article/10.1234/example"
    │
    ├─ [If DOI found] get_oa_url_for_doi(doi)
    │   └─ OpenAlex lookup → OA URL or doi.org fallback
    │
    ├─ [If PDF URL] _handle_pdf_url()
    │   ├─ httpx download
    │   ├─ Playwright fallback (anti-bot)
    │   └─ Marker processing
    │
    ├─ [Web page] scraper_service.scrape()
    │   ├─ Local Firecrawl
    │   ├─ Cloud Stealth
    │   └─ Playwright fallback
    │
    ├─ classify_content(markdown)
    │   ├─ Quick heuristics (paywall phrases, DOI errors)
    │   ├─ Article structure detection (>20k chars + headers)
    │   └─ LLM classification (DeepSeek V3)
    │
    └─ [If paywall] try_retrieve_academic(doi)
        ├─ Health check
        ├─ Submit retrieval job
        ├─ Poll for completion
        └─ Process via Marker
```

## Implementation

### Step 1: Core Data Types

```python
# core/scraping/types.py

from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ContentSource(str, Enum):
    SCRAPED = "scraped"                     # Web scraping
    PDF_DIRECT = "pdf_direct"               # Direct PDF download
    PDF_EXTRACTED = "pdf_extracted"         # PDF from abstract page
    RETRIEVE_ACADEMIC = "retrieve_academic"  # Fallback service

class ContentClassification(str, Enum):
    FULL_TEXT = "full_text"                 # Complete article
    ABSTRACT_WITH_PDF = "abstract_with_pdf"  # Abstract + PDF link
    PAYWALL = "paywall"                     # Access restricted
    NON_ACADEMIC = "non_academic"           # Not academic content

class DoiInfo(BaseModel):
    doi: str                                # Normalized DOI
    doi_url: str                            # https://doi.org/...
    source: str                             # "input", "url", "content", "title_search"

class GetUrlResult(BaseModel):
    url: str                                # Original URL/DOI
    resolved_url: Optional[str] = None      # Final URL after resolution
    content: str                            # Markdown content
    content_type: str = "markdown"
    source: ContentSource
    provider: str                           # e.g., "marker", "firecrawl-local"
    doi: Optional[str] = None
    classification: Optional[ContentClassification] = None
    links: list[str] = Field(default_factory=list)
    fallback_chain: list[str] = Field(default_factory=list)

class GetUrlOptions(BaseModel):
    pdf_quality: str = "balanced"           # fast, balanced, quality
    pdf_langs: list[str] = ["English"]
    detect_academic: bool = True
    allow_retrieve_academic: bool = True
    include_links: bool = True
    scrape_timeout: float = 60.0
    retrieve_academic_timeout: float = 180.0
```

### Step 2: DOI Detection

```python
# core/scraping/doi/detector.py

import re
from typing import Optional

DOI_REGEX = re.compile(r"10\.\d{4,}/[^\s<>\"'\]\),;]+")

# Publisher-specific URL patterns
PUBLISHER_PATTERNS = [
    (r"link\.springer\.com/article/(10\.\d+/.+)", "springer"),
    (r"onlinelibrary\.wiley\.com/doi/(10\.\d+/.+)", "wiley"),
    (r"pnas\.org/doi/(10\.\d+/.+)", "pnas"),
    (r"academic\.oup\.com/.*/doi/(10\.\d+/.+)", "oxford"),
    (r"tandfonline\.com/doi/(?:full|abs)/(10\.\d+/.+)", "taylor_francis"),
    (r"journals\.sagepub\.com/doi/(10\.\d+/.+)", "sage"),
    (r"pubs\.acs\.org/doi/(10\.\d+/.+)", "acs"),
    (r"iopscience\.iop\.org/article/(10\.\d+/.+)", "iop"),
    (r"cambridge\.org/core/.*/doi/(10\.\d+/.+)", "cambridge"),
]


def detect_doi(url_or_doi: str) -> Optional[DoiInfo]:
    """Detect DOI from input (bare DOI, doi.org URL, or publisher URL)."""
    text = url_or_doi.strip()

    # Bare DOI: "10.1234/example"
    if text.startswith("10."):
        match = DOI_REGEX.match(text)
        if match:
            doi = _normalize_doi(match.group())
            return DoiInfo(doi=doi, doi_url=f"https://doi.org/{doi}", source="input")

    # DOI URL: "https://doi.org/10.1234/example"
    if "doi.org/" in text:
        match = DOI_REGEX.search(text)
        if match:
            doi = _normalize_doi(match.group())
            return DoiInfo(doi=doi, doi_url=f"https://doi.org/{doi}", source="url")

    # Publisher-specific patterns
    for pattern, source in PUBLISHER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = _normalize_doi(match.group(1))
            return DoiInfo(doi=doi, doi_url=f"https://doi.org/{doi}", source=source)

    return None


def _normalize_doi(doi: str) -> str:
    """Remove trailing punctuation."""
    return doi.rstrip(".,;:")
```

### Step 3: OpenAlex DOI Resolution

```python
# core/scraping/doi/resolver.py

import httpx
from typing import Optional

OPENALEX_BASE = "https://api.openalex.org"


async def get_oa_url_for_doi(doi: str) -> Optional[str]:
    """Get best open access URL for DOI via OpenAlex.

    Priority:
    1. Primary location PDF URL
    2. Primary location landing page
    3. Best OA location PDF
    4. Best OA location landing page
    5. Any OA location PDF
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{OPENALEX_BASE}/works/doi:{doi}",
            headers=_get_headers(),
        )

        if response.status_code != 200:
            return None

        work = response.json()

        # Priority 1: Primary location PDF
        primary = work.get("primary_location", {})
        if primary.get("pdf_url"):
            return primary["pdf_url"]

        # Priority 2: Primary landing page
        if primary.get("landing_page_url"):
            return primary["landing_page_url"]

        # Priority 3-5: OA locations
        best_oa = work.get("best_oa_location", {})
        if best_oa.get("pdf_url"):
            return best_oa["pdf_url"]
        if best_oa.get("landing_page_url"):
            return best_oa["landing_page_url"]

        # Any OA PDF
        for loc in work.get("open_access", {}).get("oa_locations", []):
            if loc.get("pdf_url"):
                return loc["pdf_url"]

        return None


async def search_doi_by_title(title: str, authors: Optional[list[str]] = None) -> Optional[str]:
    """Search OpenAlex for DOI by title when paywall detected."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{OPENALEX_BASE}/works",
            params={"search": title, "per_page": 1},
            headers=_get_headers(),
        )

        if response.status_code != 200:
            return None

        results = response.json().get("results", [])
        if not results:
            return None

        work = results[0]
        work_title = work.get("title", "")

        # Require >80% title similarity
        if _title_similarity(title, work_title) < 0.8:
            return None

        return work.get("doi", "").replace("https://doi.org/", "")
```

### Step 4: Content Classification

```python
# core/scraping/classification/classifier.py

from typing import Optional
from pydantic import BaseModel, Field

PAYWALL_INDICATORS = [
    "sign in to access", "sign in to view",
    "subscribe to read", "purchase this article",
    "institutional access required", "access denied",
    "you do not have access", "rent or purchase",
    "buy this article", "get full access",
    "login required", "members only",
]

ARTICLE_SECTIONS = [
    "introduction", "methods", "results", "discussion",
    "conclusion", "background", "abstract", "references",
]


class ClassificationResult(BaseModel):
    classification: str  # full_text, abstract_with_pdf, paywall, non_academic
    confidence: float = Field(ge=0.0, le=1.0)
    pdf_url: Optional[str] = None
    reasoning: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None


async def classify_content(
    markdown: str,
    url: str,
    doi: Optional[str] = None,
    links: Optional[list[str]] = None,
) -> ClassificationResult:
    """Classify content using heuristics and LLM fallback."""

    # Tier 1: Quick paywall detection
    if _quick_paywall_check(markdown):
        return ClassificationResult(
            classification="paywall",
            confidence=0.95,
            reasoning="Detected paywall indicators in content",
        )

    # Tier 2: DOI error page detection
    if _is_doi_error_page(markdown):
        return ClassificationResult(
            classification="paywall",
            confidence=0.95,
            reasoning="DOI resolver error page detected",
        )

    # Tier 3: Quick full text detection
    if len(markdown) > 20000 and _has_article_structure(markdown):
        return ClassificationResult(
            classification="full_text",
            confidence=0.9,
            reasoning="Long content with article structure",
        )

    # Tier 4: LLM classification for ambiguous cases
    return await _llm_classify(markdown, url, doi, links)


def _quick_paywall_check(markdown: str) -> bool:
    """Check for paywall indicators."""
    lower = markdown.lower()
    return any(ind in lower for ind in PAYWALL_INDICATORS)


def _has_article_structure(markdown: str) -> bool:
    """Check for typical article section headers."""
    lower = markdown.lower()
    found = sum(1 for section in ARTICLE_SECTIONS if section in lower)
    return found >= 3
```

### Step 5: Academic Fallback

```python
# core/scraping/fallback/academic.py

from typing import Optional
from core.stores.retrieve_academic import (
    retrieve_academic_available,
    submit_retrieval,
    wait_for_job,
    download_to_temp,
)
from core.scraping.pdf import process_pdf_file
from core.scraping.types import ContentSource, GetUrlResult


async def try_retrieve_academic(
    doi: str,
    timeout: float = 180.0,
    fallback_chain: Optional[list[str]] = None,
) -> Optional[GetUrlResult]:
    """Try retrieve-academic fallback for paywalled content."""
    chain = fallback_chain or []

    # Health check
    if not await retrieve_academic_available():
        logger.warning("retrieve-academic service unavailable")
        return None

    chain.append("retrieve_academic")

    # Submit and wait
    job_id = await submit_retrieval(doi, timeout=timeout)
    if not job_id:
        return None

    result = await wait_for_job(job_id, timeout=timeout)
    if result.get("status") != "completed":
        logger.error(f"Retrieval failed: {result.get('error_message')}")
        return None

    # Download and process
    local_path = await download_to_temp(job_id, doi)
    markdown = await process_pdf_file(local_path)

    return GetUrlResult(
        url=f"https://doi.org/{doi}",
        resolved_url=local_path.as_uri(),
        content=markdown,
        source=ContentSource.RETRIEVE_ACADEMIC,
        provider="retrieve-academic",
        doi=doi,
        fallback_chain=chain,
    )
```

### Step 6: Main Entry Point

```python
# core/scraping/unified.py

from typing import Optional
from core.scraping.types import (
    ContentSource, GetUrlOptions, GetUrlResult, DoiInfo
)
from core.scraping.doi import detect_doi, get_oa_url_for_doi, search_doi_by_title
from core.scraping.pdf import is_pdf_url, process_pdf_url
from core.scraping.classification import classify_content
from core.scraping.fallback import try_retrieve_academic
from core.scraping.service import get_scraper_service


async def get_url(
    url: str,
    options: Optional[GetUrlOptions] = None,
) -> GetUrlResult:
    """Unified content retrieval with DOI resolution and academic fallback.

    Flow:
    1. DOI detection → OpenAlex OA URL resolution
    2. PDF detection → Marker processing
    3. Web scraping → Firecrawl/Playwright chain
    4. Content classification → Heuristics + LLM
    5. Academic fallback → retrieve-academic service
    """
    opts = options or GetUrlOptions()
    fallback_chain: list[str] = []
    doi_info: Optional[DoiInfo] = None
    resolved_url = url

    # === Step 1: DOI Detection ===
    doi_info = detect_doi(url)
    if doi_info:
        fallback_chain.append("doi_detected")

        # Try to get OA URL
        oa_url = await get_oa_url_for_doi(doi_info.doi)
        if oa_url:
            resolved_url = oa_url
            fallback_chain.append("openalex_oa_url")
        else:
            resolved_url = doi_info.doi_url
            fallback_chain.append("doi_url_fallback")

    # === Step 2: PDF Detection ===
    if is_pdf_url(resolved_url):
        fallback_chain.append("pdf_detected")
        result = await _handle_pdf_url(resolved_url, opts, fallback_chain)
        if result:
            result.doi = doi_info.doi if doi_info else None
            return result

    # === Step 3: Web Scraping ===
    scraper = get_scraper_service()
    scrape_result = await scraper.scrape(resolved_url, include_links=opts.include_links)
    fallback_chain.append(f"scraper:{scrape_result.provider}")

    # === Step 4: Content Classification ===
    if opts.detect_academic:
        classification = await classify_content(
            scrape_result.markdown,
            resolved_url,
            doi_info.doi if doi_info else None,
            scrape_result.links,
        )
        fallback_chain.append(f"classified:{classification.classification}")

        # Handle abstract_with_pdf: Extract PDF and process
        if classification.classification == "abstract_with_pdf" and classification.pdf_url:
            pdf_result = await _handle_pdf_url(classification.pdf_url, opts, fallback_chain)
            if pdf_result:
                pdf_result.doi = doi_info.doi if doi_info else None
                pdf_result.classification = classification.classification
                return pdf_result

        # Handle paywall: Try title search if no DOI
        if classification.classification == "paywall":
            fallback_chain.append("paywall_detected")

            if not doi_info and classification.title:
                searched_doi = await search_doi_by_title(
                    classification.title, classification.authors
                )
                if searched_doi:
                    doi_info = DoiInfo(
                        doi=searched_doi,
                        doi_url=f"https://doi.org/{searched_doi}",
                        source="title_search",
                    )
                    fallback_chain.append("doi_from_title_search")

    # === Step 5: Academic Fallback ===
    if (
        opts.allow_retrieve_academic
        and doi_info
        and classification
        and classification.classification == "paywall"
    ):
        academic_result = await try_retrieve_academic(
            doi_info.doi,
            timeout=opts.retrieve_academic_timeout,
            fallback_chain=fallback_chain,
        )
        if academic_result:
            return academic_result

    # Return scraped content
    return GetUrlResult(
        url=url,
        resolved_url=resolved_url,
        content=scrape_result.markdown,
        source=ContentSource.SCRAPED,
        provider=scrape_result.provider,
        doi=doi_info.doi if doi_info else None,
        classification=classification.classification if classification else None,
        links=scrape_result.links,
        fallback_chain=fallback_chain,
    )
```

## Configuration

Environment variables:

```bash
# OpenAlex (optional but recommended)
OPENALEX_EMAIL=your@email.com           # Polite pool for better rate limits

# Firecrawl (see unified-scraping-service-fallback-chain.md)
FIRECRAWL_LOCAL_URL=http://localhost:3002
FIRECRAWL_API_KEY=fc-xxxxx

# Marker PDF processing
MARKER_BASE_URL=http://localhost:8001
MARKER_INPUT_DIR=/data/input

# retrieve-academic (requires VPN to institutional network)
RETRIEVE_ACADEMIC_URL=http://retrieve-academic:8000
```

## Consequences

### Benefits

- **Single entry point**: `get_url()` handles all content types uniformly
- **DOI intelligence**: Automatic detection from bare DOIs, URLs, and publisher patterns
- **Open access priority**: OpenAlex lookup finds OA versions before hitting paywalls
- **Graceful degradation**: 5-level fallback ensures maximum content acquisition
- **Audit trail**: `fallback_chain` tracks exactly how content was acquired
- **Title-based DOI search**: Can find DOI even when starting from a URL

### Trade-offs

- **Complexity**: 5-step pipeline with multiple failure modes
- **External dependencies**: Requires OpenAlex, Marker, optionally retrieve-academic
- **Classification latency**: LLM classification adds ~2-3s for ambiguous content
- **VPN requirement**: retrieve-academic fallback needs institutional network access

## Related Patterns

- [Unified Scraping Service with Fallback Chain](./unified-scraping-service-fallback-chain.md) - Low-level scraping tier
- [Citation Processing with Zotero Integration](./citation-processing-zotero-integration.md) - Downstream citation handling
- [GPU-Accelerated Document Processing](./gpu-accelerated-document-processing-service.md) - Marker PDF service
- [Hash-Based Persistent Caching](./hash-based-persistent-caching.md) - OpenAlex/Marker result caching

## Known Uses

- `core/scraping/unified.py`: Main `get_url()` implementation
- `core/scraping/doi/detector.py`: DOI detection patterns
- `core/scraping/doi/resolver.py`: OpenAlex integration
- `core/scraping/classification/classifier.py`: Content classification
- `core/scraping/fallback/academic.py`: retrieve-academic integration
- `core/scraping/pdf/processor.py`: Marker PDF processing
- `core/scraping/types.py`: Data models

## References

- [OpenAlex API Documentation](https://docs.openalex.org/)
- [DOI Handbook](https://www.doi.org/doi_handbook/2_Numbering.html)
- [CrossRef DOI Display Guidelines](https://www.crossref.org/display-guidelines/)
