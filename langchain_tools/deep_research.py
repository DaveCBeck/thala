"""
Deep research LangChain tool.

Provides a tool wrapper for the deep research workflow, enabling
use via MCP server and LangChain agents.
"""

import logging
from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeepResearchOutput(BaseModel):
    """Output schema for deep_research tool."""

    success: bool = Field(description="Whether the research completed successfully")
    topic: str = Field(default="", description="The research topic")
    final_report: Optional[str] = Field(default=None, description="The final research report")
    citations: list[dict] = Field(default_factory=list, description="Sources cited in the report")
    store_record_id: Optional[str] = Field(default=None, description="UUID of saved record in store")
    iterations: int = Field(default=0, description="Number of research iterations completed")
    sources_consulted: int = Field(default=0, description="Number of web sources consulted")
    memory_findings_used: int = Field(default=0, description="Number of memory findings incorporated")
    status: str = Field(default="unknown", description="Final status of the research")
    errors: list[dict] = Field(default_factory=list, description="Any errors encountered")


@tool
async def deep_research(
    query: str,
    depth: Literal["quick", "standard", "comprehensive"] = "standard",
    max_sources: int = 20,
) -> dict:
    """Conduct deep research on a topic with comprehensive web search.

    Use this when you need to:
    - Research a complex topic thoroughly
    - Gather information from multiple web sources
    - Produce a comprehensive report with citations
    - Integrate findings with existing knowledge

    The workflow:
    1. Clarifies the research question if ambiguous
    2. Searches your memory for relevant existing knowledge
    3. Customizes the research plan based on what you already know
    4. Conducts iterative web research with multiple search queries
    5. Synthesizes findings into a comprehensive report
    6. Saves the research to your knowledge base

    Depth options:
    - "quick": 2 iterations, ~5 minutes, good for focused questions
    - "standard": 4 iterations, ~15 minutes, balanced depth
    - "comprehensive": 8+ iterations, 30+ minutes, exhaustive research

    Args:
        query: The research question or topic to investigate
        depth: Research depth ("quick", "standard", "comprehensive")
        max_sources: Maximum number of web sources to consult (default 20)

    Returns:
        Research report with citations and metadata

    Example:
        result = await deep_research(
            "What are the latest developments in quantum computing?",
            depth="standard"
        )
    """
    try:
        from workflows.research import deep_research as run_research
    except ImportError as e:
        logger.error(f"Failed to import research workflow: {e}")
        return DeepResearchOutput(
            success=False,
            status="import_error",
            errors=[{"error": str(e)}],
        ).model_dump(mode="json")

    try:
        result = await run_research(
            query=query,
            depth=depth,
            max_sources=max_sources,
        )

        # Extract key metrics
        diffusion = result.get("diffusion", {})
        findings = result.get("research_findings", [])
        memory_findings = result.get("memory_findings", [])

        output = DeepResearchOutput(
            success=result.get("current_status") == "completed",
            topic=result.get("research_brief", {}).get("topic", query),
            final_report=result.get("final_report"),
            citations=result.get("citations", []),
            store_record_id=result.get("store_record_id"),
            iterations=diffusion.get("iteration", 0),
            sources_consulted=len(findings),
            memory_findings_used=len(memory_findings),
            status=result.get("current_status", "unknown"),
            errors=result.get("errors", []),
        )

        logger.info(
            f"Deep research completed: topic='{output.topic[:30]}...', "
            f"success={output.success}, iterations={output.iterations}"
        )

        return output.model_dump(mode="json")

    except Exception as e:
        logger.exception(f"Deep research failed: {e}")
        return DeepResearchOutput(
            success=False,
            status="error",
            errors=[{"error": str(e)}],
        ).model_dump(mode="json")
