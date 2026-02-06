"""Executor registry for structured output strategies.

Used internally by invoke() for structured output extraction.
"""

from ..types import StructuredOutputStrategy
from .base import StrategyExecutor
from .langchain import LangChainStructuredExecutor
from .agent import ToolAgentExecutor
from .json import JSONPromptingExecutor


# Executor instances (singletons - no state)
executors: dict[StructuredOutputStrategy, StrategyExecutor] = {
    StructuredOutputStrategy.LANGCHAIN_STRUCTURED: LangChainStructuredExecutor(),
    StructuredOutputStrategy.TOOL_AGENT: ToolAgentExecutor(),
    StructuredOutputStrategy.JSON_PROMPTING: JSONPromptingExecutor(),
}


def get_executor(strategy: StructuredOutputStrategy) -> StrategyExecutor:
    """Get executor instance for a strategy."""
    return executors[strategy]


__all__ = [
    "StrategyExecutor",
    "LangChainStructuredExecutor",
    "ToolAgentExecutor",
    "JSONPromptingExecutor",
    "executors",
    "get_executor",
]
