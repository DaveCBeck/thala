"""Agent runner for tool-enabled LLM interactions.

Provides a simple agent loop that:
1. Invokes LLM with tools + output schema as a tool
2. Executes any tool calls
3. When "submit_result" tool is called, extracts and returns the structured output
4. Continues until submit_result or budget exceeded
"""

from .result_parser import create_submit_tool, parse_submit_result
from .runner import run_tool_agent, run_tool_agent_simple
from .types import AgentBudget

__all__ = [
    "AgentBudget",
    "create_submit_tool",
    "parse_submit_result",
    "run_tool_agent",
    "run_tool_agent_simple",
]
