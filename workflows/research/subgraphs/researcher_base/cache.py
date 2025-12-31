"""TTL cache for scraped URL content."""

import os
from cachetools import TTLCache

# Cache for scraped URL content (1 hour TTL, max 200 items)
# Shared across ALL researcher types to avoid redundant scrapes
_scrape_cache: TTLCache = TTLCache(
    maxsize=int(os.getenv("SCRAPE_CACHE_SIZE", "200")),
    ttl=int(os.getenv("SCRAPE_CACHE_TTL", "3600")),  # 1 hour default
)


def get_scrape_cache() -> TTLCache:
    """Get the shared scrape cache instance."""
    return _scrape_cache
