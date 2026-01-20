"""Tool agent executor for structured output.

Uses multi-turn agent loop with output as final tool call.
"""

from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from ...models import get_llm
from ..types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
)
from .base import StrategyExecutor

T = TypeVar("T", bound=BaseModel)


class ToolAgentExecutor(StrategyExecutor[T]):
    """Uses multi-turn agent loop with output as final tool."""

    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: str,
        system_prompt: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from .agent_runner import run_tool_agent

        llm = get_llm(
            tier=output_config.tier,
            max_tokens=output_config.max_tokens,
            thinking_budget=output_config.thinking_budget,
        )

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        result = await run_tool_agent(
            llm=llm,
            tools=output_config.tools,
            messages=messages,
            output_schema=output_schema,
            max_tool_calls=output_config.max_tool_calls,
            max_total_chars=output_config.max_tool_result_chars,
        )

        return StructuredOutputResult.ok(
            value=result,
            strategy=StructuredOutputStrategy.TOOL_AGENT,
        )


__all__ = ["ToolAgentExecutor"]
