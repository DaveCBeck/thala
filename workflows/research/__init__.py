"""
Deep Research Workflow.

Implements a Self-Balancing Diffusion Algorithm for comprehensive research:
1. Clarify user intent
2. Create research brief
3. Search Thala memory stores for existing knowledge
4. Iterate/customize plan based on memory context
5. Supervisor coordinates parallel researcher agents
6. Generate final report with citations
7. Save findings to store
"""

import logging

from workflows.research.graph import deep_research
from workflows.research.state import DeepResearchState

logger = logging.getLogger(__name__)

__all__ = ["deep_research", "DeepResearchState", "cleanup_research_resources"]


async def cleanup_research_resources() -> None:
    """Clean up all research workflow resources (idempotent)."""
    from core.scraping.service import close_scraper_service
    from langchain_tools.firecrawl import close_firecrawl

    # Close both global singletons
    try:
        await close_firecrawl()
    except Exception as e:
        logger.warning(f"Error closing Firecrawl client: {e}")

    try:
        await close_scraper_service()
    except Exception as e:
        logger.warning(f"Error closing scraper service: {e}")
