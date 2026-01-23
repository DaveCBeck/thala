"""LangChain structured output executor.

Uses LangChain's .with_structured_output() method for obtaining structured responses.
"""

from typing import Optional, Type, TypeVar

from langsmith import traceable
from pydantic import BaseModel

from ...caching import create_cached_messages, warm_deepseek_cache
from ...models import get_llm, is_deepseek_tier
from ..types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
)
from .base import StrategyExecutor

T = TypeVar("T", bound=BaseModel)


class LangChainStructuredExecutor(StrategyExecutor[T]):
    """Uses LangChain's .with_structured_output() method."""

    @traceable(run_type="llm", name="structured_output")
    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: str,
        system_prompt: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        llm = get_llm(
            tier=output_config.tier,
            max_tokens=output_config.max_tokens,
            thinking_budget=output_config.thinking_budget,
        )

        # Select structured output method based on model and config:
        # - DeepSeek: Must use function_calling (json_schema not supported)
        # - Claude with thinking: Default method (json_schema conflicts with tool_choice)
        # - Claude without thinking + use_json_schema_method: json_schema for stricter validation
        # - Default: Let LangChain choose (usually json_schema for OpenAI-compatible)
        if is_deepseek_tier(output_config.tier):
            # DeepSeek doesn't support json_schema response_format, use function calling
            structured_llm = llm.with_structured_output(
                output_schema, method="function_calling"
            )
        elif output_config.use_json_schema_method and not output_config.thinking_budget:
            structured_llm = llm.with_structured_output(
                output_schema, method="json_schema"
            )
        else:
            structured_llm = llm.with_structured_output(output_schema)

        # Build messages
        if system_prompt:
            if output_config.enable_prompt_cache:
                if is_deepseek_tier(output_config.tier):
                    # DeepSeek: standard messages with prefix-based caching
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    # Ensure cache is warmed up for this prefix
                    await warm_deepseek_cache(system_prompt)
                else:
                    # Anthropic: explicit cache_control
                    messages = create_cached_messages(
                        system_content=system_prompt,
                        user_content=user_prompt,
                        cache_ttl=output_config.cache_ttl,
                    )
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        result = await structured_llm.ainvoke(messages)

        return StructuredOutputResult.ok(
            value=result,
            strategy=StructuredOutputStrategy.LANGCHAIN_STRUCTURED,
            usage=getattr(result, "usage_metadata", None),
        )


__all__ = ["LangChainStructuredExecutor"]
