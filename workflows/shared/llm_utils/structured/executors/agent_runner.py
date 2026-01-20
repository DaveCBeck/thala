"""Simple tool agent runner for structured output.

Implements a multi-turn agent loop that executes tools and returns structured output.
"""

import logging
from typing import Type, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langsmith import traceable
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@traceable(name="tool_agent")
async def run_tool_agent(
    llm: ChatAnthropic,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    output_schema: Type[T],
    max_tool_calls: int = 12,
    max_total_chars: int = 100000,
) -> T:
    """Execute multi-turn agent loop with tools, returning structured output.

    Args:
        llm: The ChatAnthropic LLM instance
        tools: List of LangChain tools to make available
        messages: Initial messages (system + user)
        output_schema: Pydantic model for the expected output
        max_tool_calls: Maximum number of tool calls before forcing output
        max_total_chars: Maximum characters from tool results

    Returns:
        Parsed output matching output_schema
    """
    # Build tool map for execution
    tool_map = {tool.name: tool for tool in tools}

    # Bind tools to LLM (without forcing tool_choice for thinking compatibility)
    llm_with_tools = llm.bind_tools(tools)

    total_chars = 0
    call_count = 0
    working_messages = list(messages)

    while call_count < max_tool_calls:
        # Get LLM response
        response: AIMessage = await llm_with_tools.ainvoke(working_messages)
        working_messages.append(response)

        # Check for tool calls
        if not response.tool_calls:
            # No more tool calls - LLM is done with tools
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name not in tool_map:
                # Unknown tool - add error message
                tool_result = f"Error: Unknown tool '{tool_name}'"
                logger.warning(f"Unknown tool requested: {tool_name}")
            else:
                try:
                    tool = tool_map[tool_name]
                    result = await tool.ainvoke(tool_args)
                    tool_result = str(result) if result is not None else ""

                    # Truncate if needed
                    if len(tool_result) > max_total_chars - total_chars:
                        available = max(0, max_total_chars - total_chars)
                        tool_result = tool_result[:available] + "\n[truncated]"

                    total_chars += len(tool_result)
                    logger.debug(
                        f"Tool {tool_name} returned {len(tool_result)} chars"
                    )
                except Exception as e:
                    tool_result = f"Error executing tool: {e}"
                    logger.warning(f"Tool {tool_name} failed: {e}")

            # Add tool result message
            working_messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )
            call_count += 1

        # Check if we've hit limits
        if total_chars >= max_total_chars:
            logger.info(f"Reached max tool result chars ({max_total_chars})")
            break

    logger.debug(f"Agent loop completed: {call_count} tool calls, {total_chars} chars")

    # Now get structured output
    # Use with_structured_output without tool_choice forcing
    structured_llm = llm.with_structured_output(output_schema)

    # Add instruction to produce final output
    working_messages.append(
        {"role": "user", "content": "Based on the information gathered, provide your final structured response."}
    )

    result = await structured_llm.ainvoke(working_messages)

    return result


__all__ = ["run_tool_agent"]
