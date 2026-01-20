"""LangChain structured output executor.

Uses LangChain's .with_structured_output() method for obtaining structured responses.
"""

from typing import Optional, Type, TypeVar

from langsmith import traceable
from pydantic import BaseModel

from ...caching import create_cached_messages
from ...models import get_llm
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

        # json_schema method uses tool_choice which conflicts with thinking.
        # When thinking is enabled, fall back to default method and rely on retries.
        use_json_schema = output_config.use_json_schema_method and not output_config.thinking_budget
        if use_json_schema:
            structured_llm = llm.with_structured_output(
                output_schema, method="json_schema"
            )
        else:
            structured_llm = llm.with_structured_output(output_schema)

        # Build messages
        if system_prompt:
            if output_config.enable_prompt_cache:
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
