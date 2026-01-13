"""Unified scraping service with Firecrawl + Playwright fallback.

Fallback chain:
1. Local Firecrawl (self-hosted) - fast, no rate limits
2. Cloud Firecrawl stealth - for anti-bot bypass
3. Playwright (local browser) - for blocklisted sites

Usage:
    from core.scraping import get_scraper_service

    service = get_scraper_service()
    result = await service.scrape("https://example.com")
    print(result.markdown)
    print(result.provider)  # "firecrawl-local", "firecrawl-stealth", or "playwright"
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

__all__ = [
    # Main service
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
