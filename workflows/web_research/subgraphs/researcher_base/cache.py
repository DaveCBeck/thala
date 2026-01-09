"""TTL cache for scraped URL content."""

from workflows.shared import get_ttl_cache

# Backwards-compatible: same env vars (SCRAPE_CACHE_SIZE, SCRAPE_CACHE_TTL)
_scrape_cache = get_ttl_cache("scrape")


def get_scrape_cache():
    """Get the shared scrape cache instance."""
    return _scrape_cache
