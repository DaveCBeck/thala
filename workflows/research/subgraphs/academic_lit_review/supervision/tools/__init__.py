"""LLM tools for supervision loops.

Provides tools that LLMs can call during editing and fact-checking:
- search_papers: Hybrid search for papers by topic
- get_paper_content: Fetch detailed L2 content for a paper
"""

from .paper_search import create_paper_tools
from .agent_runner import run_tool_agent

__all__ = [
    "create_paper_tools",
    "run_tool_agent",
]
