"""Custom exceptions for scraping service."""


class ScrapingError(Exception):
    """Base scraping exception."""

    def __init__(self, message: str, url: str, provider: str | None = None):
        self.message = message
        self.url = url
        self.provider = provider
        super().__init__(message)


class SiteBlockedError(ScrapingError):
    """Site is blocked by the scraping provider (e.g., Firecrawl blocklist)."""

    pass


class ScrapingTimeoutError(ScrapingError):
    """Scraping operation timed out."""

    pass


class LocalServiceUnavailableError(ScrapingError):
    """Local Firecrawl service is unavailable (not running or unreachable)."""

    pass
