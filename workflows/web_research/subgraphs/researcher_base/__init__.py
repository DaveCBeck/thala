"""
Shared utilities for specialized researcher subgraphs.

Contains:
- PDF processing via MD5 retrieval (Anna's Archive pattern)
- URL scraping helper (uses core.scraping.get_url)
- Query validation
- Query generation (identical for all researcher types)
"""

from .pdf_processor import fetch_pdf_via_marker
from .url_scraper import scrape_single_url, scrape_pages
from .query_validator import validate_queries
from .query_generator import create_generate_queries, generate_queries, RESEARCHER_QUERY_PROMPTS

__all__ = [
    "fetch_pdf_via_marker",
    "scrape_single_url",
    "scrape_pages",
    "validate_queries",
    "create_generate_queries",
    "generate_queries",
    "RESEARCHER_QUERY_PROMPTS",
]
