"""Unified URL content retrieval system.

Primary interface:
    from core.scraping import get_url, GetUrlResult, GetUrlOptions

    # Simple usage - handles URLs, DOIs, PDFs automatically
    result = await get_url("https://example.com")
    result = await get_url("10.1038/nature12373")  # DOI
    result = await get_url("https://arxiv.org/pdf/2301.00001.pdf")  # PDF

    # With options
    result = await get_url(url, GetUrlOptions(pdf_quality="quality"))

Low-level scraping service (for direct control):
    from core.scraping import get_scraper_service

    service = get_scraper_service()
    result = await service.scrape("https://example.com")

Fallback chain:
1. DOI resolution via OpenAlex (if DOI detected)
2. PDF processing via Marker (if PDF URL)
3. Web scraping: Local Firecrawl → Cloud Stealth → Playwright
4. Content classification (academic detection)
5. retrieve-academic fallback (for paywalled academic content)
"""

from .config import FirecrawlConfig, get_firecrawl_config
from .errors import (
    LocalServiceUnavailableError,
    ScrapingError,
    ScrapingTimeoutError,
    SiteBlockedError,
)
from .firecrawl_clients import FirecrawlClients, get_firecrawl_clients
from .playwright_scraper import PlaywrightScraper
from .service import (
    ScrapeResult,
    ScraperService,
    close_scraper_service,
    get_scraper_service,
)

# Unified URL retrieval system
from .types import (
    ContentClassification,
    ContentSource,
    DoiInfo,
    GetUrlOptions,
    GetUrlResult,
)
from .unified import get_url

__all__ = [
    # Primary interface (unified URL retrieval)
    "get_url",
    "GetUrlResult",
    "GetUrlOptions",
    "ContentSource",
    "ContentClassification",
    "DoiInfo",
    # Low-level scraping service
    "ScraperService",
    "ScrapeResult",
    "get_scraper_service",
    "close_scraper_service",
    # Configuration
    "FirecrawlConfig",
    "get_firecrawl_config",
    # Client management
    "FirecrawlClients",
    "get_firecrawl_clients",
    # Errors
    "ScrapingError",
    "SiteBlockedError",
    "ScrapingTimeoutError",
    "LocalServiceUnavailableError",
    # Low-level (for direct use if needed)
    "PlaywrightScraper",
]
