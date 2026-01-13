"""
Shared utilities for specialized researcher subgraphs.

Contains:
- URL scraping helper (uses core.scraping.get_url)
- Query validation
- Query generation (identical for all researcher types)

For MD5-based PDF retrieval, use core.scraping.process_pdf_by_md5.
"""

from .url_scraper import scrape_single_url, scrape_pages
from .query_validator import validate_queries
from .query_generator import create_generate_queries, generate_queries, RESEARCHER_QUERY_PROMPTS

__all__ = [
    "scrape_single_url",
    "scrape_pages",
    "validate_queries",
    "create_generate_queries",
    "generate_queries",
    "RESEARCHER_QUERY_PROMPTS",
]
