"""Researcher subagent — Opus with web search/scrape tools via MCP."""

import json
import logging
import os
import sys

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.cli_backend import invoke_tool_agent_via_cli

from .prompts import RESEARCHER_SYSTEM, get_today

logger = logging.getLogger(__name__)


class ResearchFinding(BaseModel):
    """Structured output from a researcher subagent."""

    finding: str = Field(description="2-4 paragraph summary with inline [N] citations")
    sources: list[dict] = Field(
        description='List of {"url": "...", "title": "...", "date": "..."} for each cited source'
    )
    confidence: str = Field(description="high, medium, or low")
    gaps: list[str] = Field(description="Remaining questions or uncertainties")


# Thin LangChain tool wrappers so invoke_tool_agent_via_cli can build the MCP config.
# The actual execution happens in the MCP server, not here.

class _WebSearch(BaseTool):
    name: str = "web_search"
    description: str = "Search the web via Firecrawl"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Handled by MCP server")

    def _run(self, query: str) -> str:
        raise NotImplementedError("Handled by MCP server")


class _PerplexitySearch(BaseTool):
    name: str = "perplexity_search"
    description: str = "AI-powered web search via Perplexity"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Handled by MCP server")

    def _run(self, query: str) -> str:
        raise NotImplementedError("Handled by MCP server")


class _ScrapeUrl(BaseTool):
    name: str = "scrape_url"
    description: str = "Fetch full text content of a web page"

    async def _arun(self, url: str) -> str:
        raise NotImplementedError("Handled by MCP server")

    def _run(self, url: str) -> str:
        raise NotImplementedError("Handled by MCP server")


_MCP_TOOLS: list[BaseTool] = [_WebSearch(), _PerplexitySearch(), _ScrapeUrl()]

# MCP tool names for the web research server
_MCP_SUPPORTED_TOOLS = {"web_search", "perplexity_search", "scrape_url"}


def _build_mcp_config() -> str:
    """Build MCP config JSON for the web research tools server."""
    mcp_env = {
        k: v for k, v in os.environ.items()
        if k.startswith("THALA_") or k.startswith("FIRECRAWL_")
        or k.startswith("PERPLEXITY_") or k == "LANGSMITH_API_KEY"
        or k.startswith("EMBEDDING_")
    }
    mcp_env["THALA_MCP_TOOLS"] = ",".join(sorted(_MCP_SUPPORTED_TOOLS))

    config = {
        "mcpServers": {
            "web_research_tools": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.web_research_tools"],
                "env": mcp_env,
            }
        }
    }
    return json.dumps(config)


async def run_researcher(
    question: str,
    context: str,
    search_hints: list[str] | None = None,
    recency_info: str = "",
    max_turns: int = 20,
) -> ResearchFinding:
    """Run a single researcher subagent via claude -p with MCP tools.

    Returns structured findings.
    """
    system = RESEARCHER_SYSTEM.format(date=get_today())

    user_parts = [f"## Research Question\n{question}"]
    if context:
        user_parts.append(f"\n## Context\n{context}")
    if search_hints:
        user_parts.append(f"\n## Suggested Search Terms\n" + ", ".join(search_hints))
    if recency_info:
        user_parts.append(f"\n## Recency\n{recency_info}")

    user_prompt = "\n".join(user_parts)

    return await _invoke_researcher(system, user_prompt, max_turns)


async def _invoke_researcher(system: str, user_prompt: str, max_turns: int = 20) -> ResearchFinding:
    """Invoke the researcher via claude -p with web research MCP tools."""
    import asyncio

    from workflows.shared.llm_utils.cli_backend import (
        _build_base_cmd,
        _run_claude_cli,
        _CLI_MAX_RETRIES,
        _RateLimitError,
        _sleep_for_rate_limit,
    )

    model = "opus"
    mcp_config = _build_mcp_config()
    json_schema = json.dumps(ResearchFinding.model_json_schema())

    # max_turns from quality tier config

    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--system-prompt", system,
        "--model", model,
        "--max-turns", str(max_turns),
        "--no-session-persistence",
        "--dangerously-skip-permissions",
        "--mcp-config", mcp_config,
        "--strict-mcp-config",
        "--json-schema", json_schema,
    ]

    last_err = None
    attempt = 0
    while attempt < _CLI_MAX_RETRIES:
        try:
            logger.debug(
                "Researcher: invoking opus (max_turns=%d, attempt=%d)",
                max_turns, attempt + 1,
            )
            envelope = await _run_claude_cli(cmd, user_prompt)
            raw = envelope["structured_output"]
            return ResearchFinding.model_validate(raw)
        except _RateLimitError as e:
            await _sleep_for_rate_limit(e, "researcher")
            continue
        except (TimeoutError, RuntimeError, KeyError) as e:
            last_err = e
            attempt += 1
            logger.warning("Researcher: attempt %d/%d failed: %s", attempt, _CLI_MAX_RETRIES, e)
            if attempt < _CLI_MAX_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))

    raise last_err  # type: ignore[misc]
