"""Unified scraping service with Firecrawl + Playwright fallback.

Usage:
    from core.scraping import get_scraper_service

    service = get_scraper_service()
    result = await service.scrape("https://example.com")
    print(result.markdown)
    print(result.provider)  # "firecrawl", "firecrawl-stealth", or "playwright"
"""

from .errors import ScrapingError, ScrapingTimeoutError, SiteBlockedError
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
    # Errors
    "ScrapingError",
    "SiteBlockedError",
    "ScrapingTimeoutError",
    # Low-level (for direct use if needed)
    "PlaywrightScraper",
]
