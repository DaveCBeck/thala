"""Broker-routed tool agent runner for structured output.

Routes each LLM round-trip through the broker for batch/cost optimization.
Uses Anthropic-native message format instead of LangChain types.
"""

import json
import logging
from typing import Any, Type, TypeVar

from langchain_core.tools import BaseTool
from langsmith import traceable
from pydantic import BaseModel

from core.llm_broker import get_broker
from core.llm_broker.schemas import BatchPolicy, LLMResponse

from .base import coerce_to_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _langchain_tool_to_anthropic(tool: BaseTool) -> dict[str, Any]:
    """Convert a LangChain BaseTool to Anthropic tool definition."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": (
            tool.args_schema.model_json_schema()
            if tool.args_schema
            else {"type": "object", "properties": {}}
        ),
    }


def _schema_to_anthropic_tool(schema: Type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic schema to Anthropic tool definition for structured output."""
    return {
        "name": schema.__name__,
        "description": f"Return the {schema.__name__} result.",
        "input_schema": schema.model_json_schema(),
    }


@traceable(run_type="tool", name="execute_tool_call_broker")
async def _execute_tool_call(
    tool: BaseTool,
    tool_name: str,
    tool_args: dict,
) -> str:
    """Execute a single tool call with LangSmith tracing."""
    result = await tool.ainvoke(tool_args)
    return str(result) if result is not None else ""


def _build_cache_control(cache_ttl: str) -> dict[str, str]:
    """Build cache_control block for the given TTL."""
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"
    return cache_control


def _add_cache_breakpoint(messages: list[dict], cache_ttl: str) -> None:
    """Add cache_control to last content block of last message.

    This creates a cache breakpoint so subsequent broker calls can reuse
    the cached prefix (tools + system + all messages up to this point).
    """
    if not messages:
        return
    cache_control = _build_cache_control(cache_ttl)
    last_msg = messages[-1]
    content = last_msg.get("content")
    if isinstance(content, list) and content:
        content[-1] = {**content[-1], "cache_control": cache_control}
    elif isinstance(content, str):
        last_msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_control}
        ]


@traceable(name="tool_agent_via_broker")
async def run_tool_agent_via_broker(
    tools: list[BaseTool],
    system_prompt: str | None,
    user_prompt: str,
    output_schema: Type[T],
    tier: Any,
    policy: BatchPolicy,
    max_tokens: int = 4096,
    effort: str | None = None,
    max_tool_calls: int = 12,
    max_total_chars: int = 100000,
    cache_ttl: str | None = None,
) -> T:
    """Execute multi-turn agent loop via broker, returning structured output.

    Each LLM round-trip goes through broker.request() with the full
    conversation history as messages. With PREFER_SPEED, calls only
    batch in ECONOMICAL mode (each round-trip waits for resolution).

    When cache_ttl is set, applies Anthropic prompt caching breakpoints:
    1. On the last tool definition (caches all tools in prefix)
    2. On the last message before each round 2+ broker call (caches
       the full conversation prefix: tools + system + messages)

    Args:
        tools: LangChain tools to make available
        system_prompt: Optional system prompt
        user_prompt: Initial user message
        output_schema: Pydantic model for expected output
        tier: Model tier (ModelTier enum)
        policy: Batch policy for broker routing
        max_tokens: Maximum output tokens per turn
        effort: Optional adaptive thinking effort level
        max_tool_calls: Maximum tool calls before forcing output
        max_total_chars: Maximum characters from tool results
        cache_ttl: Prompt cache TTL ("5m" or "1h"). None to disable.

    Returns:
        Parsed output matching output_schema
    """
    from ...invoke import _trace_batch_llm_result

    broker = get_broker()
    logger.info(
        f"run_tool_agent_via_broker: starting — tier={tier}, policy={policy}, "
        f"tools={[t.name for t in tools]}, cache_ttl={cache_ttl}, "
        f"broker_started={broker._started}"
    )
    tool_map = {tool.name: tool for tool in tools}
    anthropic_tools = [_langchain_tool_to_anthropic(tool) for tool in tools]

    # Cache breakpoint 1: mark last tool definition to cache all tools in prefix
    if cache_ttl and anthropic_tools:
        anthropic_tools[-1]["cache_control"] = _build_cache_control(cache_ttl)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    total_chars = 0
    call_count = 0
    round_number = 0

    while call_count < max_tool_calls:
        is_final_iteration = (call_count == max_tool_calls - 1) or (
            total_chars >= max_total_chars * 0.9
        )

        if is_final_iteration and call_count > 0:
            messages.append({
                "role": "user",
                "content": (
                    "You have used most of your available tool calls. "
                    "Please complete your research now and prepare to "
                    "provide your final response. You may make one more "
                    "tool call if essential, but prioritize completion."
                ),
            })

        # Cache breakpoint 2: on round 2+, mark last message to cache conversation prefix
        if cache_ttl and round_number > 0:
            _add_cache_breakpoint(messages, cache_ttl)

        logger.info(f"run_tool_agent_via_broker: round {round_number}, submitting broker request")
        future = await broker.request(
            prompt="",
            model=tier,
            policy=policy,
            max_tokens=max_tokens,
            system=system_prompt,
            effort=effort,
            tools=anthropic_tools,
            messages=messages,
        )
        # Flush immediately so the request doesn't sit in queue waiting for
        # batch threshold. Using flush() instead of batch_group() avoids
        # file lock contention when many sections run concurrently.
        await broker.flush()
        logger.info(f"run_tool_agent_via_broker: round {round_number}, flushed, awaiting future")

        response: LLMResponse = await future
        logger.info(f"run_tool_agent_via_broker: round {round_number}, got response success={response.success}")
        if response.batched and response.usage:
            await _trace_batch_llm_result(
                model=response.model or tier.value,
                usage=response.usage,
            )
        if not response.success:
            raise RuntimeError(f"Broker tool agent request failed: {response.error}")

        content_blocks = response.content_blocks or []

        # Add assistant response to conversation
        messages.append({
            "role": "assistant",
            "content": content_blocks,
        })

        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            break

        # Execute tools and build tool_result messages
        tool_results: list[dict[str, Any]] = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block["name"]
            tool_args = tool_block.get("input", {})
            tool_id = tool_block["id"]

            if tool_name not in tool_map:
                tool_result = f"Error: Unknown tool '{tool_name}'"
                logger.warning(f"Unknown tool requested: {tool_name}")
            else:
                try:
                    tool = tool_map[tool_name]
                    tool_result = await _execute_tool_call(tool, tool_name, tool_args)

                    if len(tool_result) > max_total_chars - total_chars:
                        available = max(0, max_total_chars - total_chars)
                        tool_result = tool_result[:available] + "\n[truncated]"

                    total_chars += len(tool_result)
                    logger.debug(f"Tool {tool_name} returned {len(tool_result)} chars")
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"
                    logger.warning(f"Tool {tool_name} failed: {e}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": tool_result,
            })
            call_count += 1

        messages.append({
            "role": "user",
            "content": tool_results,
        })

        round_number += 1

        if total_chars >= max_total_chars:
            logger.info(f"Reached max tool result chars ({max_total_chars})")
            break

    logger.debug(f"Agent loop completed: {call_count} tool calls, {total_chars} chars")

    # Final structured output extraction
    messages.append({
        "role": "user",
        "content": "Based on the information gathered, provide your final structured response.",
    })

    # Cache the conversation prefix before final extraction call
    if cache_ttl:
        _add_cache_breakpoint(messages, cache_ttl)

    schema_tool = _schema_to_anthropic_tool(output_schema)
    # Final extraction uses its own tool list, so cache that too
    final_tools = [schema_tool]
    if cache_ttl:
        final_tools[-1]["cache_control"] = _build_cache_control(cache_ttl)

    # Final extraction: thinking already happened in the agent loop rounds,
    # so omit effort here.  Forced tool_choice is incompatible with thinking
    # and the extraction step doesn't benefit from it anyway.
    future = await broker.request(
        prompt="",
        model=tier,
        policy=policy,
        max_tokens=max_tokens,
        system=system_prompt,
        tools=final_tools,
        tool_choice={"type": "tool", "name": output_schema.__name__},
        messages=messages,
    )
    await broker.flush()

    response = await future
    if response.batched and response.usage:
        await _trace_batch_llm_result(
            model=response.model or tier.value,
            usage=response.usage,
        )
    if not response.success:
        raise RuntimeError(f"Broker structured extraction failed: {response.error}")

    raw_data = json.loads(response.content)
    coerced = coerce_to_schema(raw_data, output_schema)
    return output_schema.model_validate(coerced)


__all__ = ["run_tool_agent_via_broker", "_add_cache_breakpoint", "_build_cache_control"]
