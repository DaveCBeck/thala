"""
Shared utilities for specialized researcher subgraphs.

Contains:
- TTL scrape cache (module-level singleton shared across all researchers)
- PDF processing via Marker
- URL scraping helper
- Query validation
- Query generation (identical for all researcher types)
"""

from .cache import get_scrape_cache
from .pdf_processor import is_pdf_url, fetch_pdf_via_marker
from .url_scraper import scrape_single_url, scrape_pages
from .query_validator import validate_queries
from .query_generator import create_generate_queries, generate_queries, RESEARCHER_QUERY_PROMPTS

__all__ = [
    "get_scrape_cache",
    "is_pdf_url",
    "fetch_pdf_via_marker",
    "scrape_single_url",
    "scrape_pages",
    "validate_queries",
    "create_generate_queries",
    "generate_queries",
    "RESEARCHER_QUERY_PROMPTS",
]
