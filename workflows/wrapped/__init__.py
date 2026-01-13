"""
Wrapped research workflow orchestrating web, academic, and book research.

Provides a unified interface for comprehensive research across multiple sources
with consistent quality settings and top_of_mind storage.
"""

from .graph.api import wrapped_research

__all__ = ["wrapped_research"]
