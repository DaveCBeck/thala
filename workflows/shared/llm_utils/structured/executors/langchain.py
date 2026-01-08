"""LangChain structured output executor.

Uses LangChain's .with_structured_output() method for obtaining structured responses.
"""

from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from ...caching import create_cached_messages
from ...models import get_llm
from ..types import StructuredOutputConfig, StructuredOutputResult, StructuredOutputStrategy
from .base import StrategyExecutor

T = TypeVar("T", bound=BaseModel)


class LangChainStructuredExecutor(StrategyExecutor[T]):
    """Uses LangChain's .with_structured_output() method."""

    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: str,
        system_prompt: Optional[str],
        config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        llm = get_llm(
            tier=config.tier,
            max_tokens=config.max_tokens,
            thinking_budget=config.thinking_budget,
        )

        if config.use_json_schema_method:
            structured_llm = llm.with_structured_output(
                output_schema, method="json_schema"
            )
        else:
            structured_llm = llm.with_structured_output(output_schema)

        # Build messages
        if system_prompt:
            if config.enable_prompt_cache:
                messages = create_cached_messages(
                    system_content=system_prompt,
                    user_content=user_prompt,
                    cache_ttl=config.cache_ttl,
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
