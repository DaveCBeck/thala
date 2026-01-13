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

from workflows.research.web_research.graph import deep_research
from workflows.research.web_research.state import DeepResearchState
from workflows.research.web_research.graph.config import QUALITY_PRESETS, WebResearchQualitySettings

logger = logging.getLogger(__name__)

__all__ = [
    "deep_research",
    "DeepResearchState",
    "cleanup_research_resources",
    "QUALITY_PRESETS",
    "WebResearchQualitySettings",
]


async def cleanup_research_resources() -> None:
    """Clean up all research workflow resources (idempotent).

    Calls the central cleanup registry which closes all HTTP clients
    that were lazily initialized during the workflow.
    """
    from core.utils.async_http_client import cleanup_all_clients
    await cleanup_all_clients()
