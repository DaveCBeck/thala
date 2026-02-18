"""Tool agent executor for structured output.

Uses multi-turn agent loop with output as final tool call.
"""

import logging
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from ...models import get_llm
from ..types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
)
from .base import StrategyExecutor, UserContent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ToolAgentExecutor(StrategyExecutor[T]):
    """Uses multi-turn agent loop with output as final tool."""

    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: UserContent,
        system_prompt: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        # Broker-routed path when batch_policy is set
        has_policy = output_config.batch_policy is not None
        is_str = isinstance(user_prompt, str)
        logger.debug(
            f"ToolAgentExecutor: batch_policy={output_config.batch_policy}, "
            f"user_prompt_is_str={is_str}, tier={output_config.tier}"
        )
        if has_policy and is_str:
            from ...models import is_deepseek_tier

            is_ds = is_deepseek_tier(output_config.tier)
            if not is_ds:
                from core.llm_broker import is_broker_enabled

                broker_on = is_broker_enabled()
                logger.info(
                    f"ToolAgentExecutor: broker routing check — "
                    f"is_deepseek={is_ds}, broker_enabled={broker_on}"
                )
                if broker_on:
                    from .broker_agent_runner import run_tool_agent_via_broker

                    logger.info("ToolAgentExecutor: routing to broker agent runner")
                    result = await run_tool_agent_via_broker(
                        tools=output_config.tools,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        output_schema=output_schema,
                        tier=output_config.tier,
                        policy=output_config.batch_policy,
                        max_tokens=output_config.max_tokens,
                        effort=output_config.effort,
                        max_tool_calls=output_config.max_tool_calls,
                        max_total_chars=output_config.max_tool_result_chars,
                        cache_ttl=output_config.cache_ttl if output_config.enable_prompt_cache else None,
                    )

                    return StructuredOutputResult.ok(
                        value=result,
                        strategy=StructuredOutputStrategy.TOOL_AGENT,
                    )

        # Existing LangChain path
        from langchain_core.messages import HumanMessage, SystemMessage

        from .agent_runner import run_tool_agent

        llm = get_llm(
            tier=output_config.tier,
            max_tokens=output_config.max_tokens,
            effort=output_config.effort,
        )

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        # HumanMessage accepts both string and list of content blocks
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
