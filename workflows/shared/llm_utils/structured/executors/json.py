"""JSON prompting executor for structured output.

Fallback strategy that prompts for JSON output and parses the response.
"""

import json
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from ...caching import create_cached_messages
from ...models import get_llm
from ...response_parsing import extract_json_from_response, extract_response_content
from ..types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
)
from .base import StrategyExecutor, coerce_to_schema

T = TypeVar("T", bound=BaseModel)


class JSONPromptingExecutor(StrategyExecutor[T]):
    """Fallback: prompts for JSON output and parses response."""

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
        )

        # Append JSON instruction
        schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
        augmented_system = (
            (system_prompt or "")
            + f"""

You must respond with ONLY valid JSON matching this schema:
{schema_json}

Do not include any text outside the JSON object. Do not wrap in markdown code blocks."""
        )

        if config.enable_prompt_cache:
            messages = create_cached_messages(
                system_content=augmented_system,
                user_content=user_prompt,
                cache_ttl=config.cache_ttl,
            )
        else:
            messages = [
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt},
            ]

        response = await llm.ainvoke(messages)
        content = extract_response_content(response)

        parsed = extract_json_from_response(content)
        coerced = coerce_to_schema(parsed, output_schema)
        validated = output_schema.model_validate(coerced)

        return StructuredOutputResult.ok(
            value=validated,
            strategy=StructuredOutputStrategy.JSON_PROMPTING,
        )


__all__ = ["JSONPromptingExecutor"]
