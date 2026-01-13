"""Agent runner for tool-enabled LLM interactions.

Provides a simple agent loop that:
1. Invokes LLM with tools + output schema as a tool
2. Executes any tool calls
3. When "submit_result" tool is called, extracts and returns the structured output
4. Continues until submit_result or budget exceeded
"""

import json
import logging
from typing import Type

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from workflows.shared.token_utils import (
    estimate_request_tokens,
    check_token_budget,
    TokenBudgetExceeded,
    HAIKU_SAFE_LIMIT,
)

from .result_parser import create_submit_tool, parse_submit_result
from .types import AgentBudget

logger = logging.getLogger(__name__)

# Message windowing configuration
MAX_MESSAGE_HISTORY = 12  # Keep system + user + last N tool interactions
SYSTEM_USER_MESSAGE_COUNT = 2  # System and initial user message
MIN_RECENT_MESSAGES = 4  # Minimum recent messages to keep


def prune_message_history(
    messages: list,
    max_history: int = MAX_MESSAGE_HISTORY,
    preserve_system_user: bool = True,
) -> list:
    """Prune message history while preserving tool_use/tool_result pairs.

    Anthropic API requires every tool_result to have a corresponding tool_use
    in the immediately preceding assistant message. This function ensures
    pruning never breaks these pairs.

    Strategy: Group messages into tool exchanges (AIMessage + ToolMessages),
    then prune at exchange boundaries.

    Args:
        messages: Full message list
        max_history: Maximum messages to keep
        preserve_system_user: Keep first system and user messages

    Returns:
        Pruned message list with tool pairs intact
    """
    if len(messages) <= max_history:
        return messages

    # Separate preserved prefix (system + user)
    if preserve_system_user and len(messages) > SYSTEM_USER_MESSAGE_COUNT:
        preserved = messages[:SYSTEM_USER_MESSAGE_COUNT]
        remaining = messages[SYSTEM_USER_MESSAGE_COUNT:]
    else:
        preserved = []
        remaining = messages

    # Group remaining messages into tool exchanges
    # An exchange = AIMessage with tool_calls + all its ToolMessage responses
    exchanges = []
    current_exchange = []

    for msg in remaining:
        if isinstance(msg, AIMessage):
            # Start of new exchange - save previous if exists
            if current_exchange:
                exchanges.append(current_exchange)
            current_exchange = [msg]
        elif isinstance(msg, ToolMessage):
            # Part of current exchange
            current_exchange.append(msg)
        else:
            # HumanMessage or other - treat as single-message exchange
            if current_exchange:
                exchanges.append(current_exchange)
                current_exchange = []
            exchanges.append([msg])

    # Don't forget last exchange
    if current_exchange:
        exchanges.append(current_exchange)

    # Calculate how many messages we can keep from exchanges
    target_messages = max_history - len(preserved)

    # Take most recent exchanges that fit within budget
    kept_exchanges = []
    message_count = 0

    for exchange in reversed(exchanges):
        if message_count + len(exchange) <= target_messages or not kept_exchanges:
            kept_exchanges.insert(0, exchange)
            message_count += len(exchange)
        else:
            break

    # Flatten exchanges back to message list
    recent = [msg for exchange in kept_exchanges for msg in exchange]

    pruned_count = len(messages) - len(preserved) - len(recent)
    if pruned_count > 0:
        logger.debug(f"Pruned {pruned_count} messages from history (kept {len(kept_exchanges)} exchanges)")

    return preserved + recent


def preflight_token_check(
    messages: list,
    has_tools: bool = True,
    max_tokens: int = HAIKU_SAFE_LIMIT,
) -> None:
    """Pre-flight check to validate request won't exceed token limits.

    Should be called before LLM invocation to fail fast.

    Args:
        messages: Message list to send
        has_tools: Whether tools are bound to LLM
        max_tokens: Token limit for model

    Raises:
        TokenBudgetExceeded: If estimated tokens exceed limit
    """
    # Estimate content from messages
    content_parts = []
    for msg in messages:
        if hasattr(msg, "content"):
            content_parts.append(str(msg.content))

    total_content = "\n".join(content_parts)

    estimated = estimate_request_tokens(
        user_prompt=total_content,
        message_count=len(messages),
        include_tool_definitions=has_tools,
    )

    check_token_budget(
        estimated_tokens=estimated,
        limit=max_tokens,
        threshold=0.90,  # 90% threshold for pre-flight
        raise_on_exceed=True,
    )


async def execute_tool(
    tool_call: dict,
    tools_by_name: dict[str, BaseTool],
) -> str:
    """Execute a single tool call.

    Args:
        tool_call: Tool call dict with 'name' and 'args'
        tools_by_name: Mapping of tool names to tool instances

    Returns:
        JSON string result from tool execution
    """
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})

    if tool_name not in tools_by_name:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        tool = tools_by_name[tool_name]
        # Tools are async, invoke them
        result = await tool.ainvoke(tool_args)

        # Convert to JSON if not already string
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)

    except Exception as e:
        logger.warning(f"Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})


async def run_tool_agent(
    llm,
    tools: list[BaseTool],
    messages: list,
    output_schema: Type[BaseModel],
    max_tool_calls: int = 12,
    max_total_chars: int = 100000,
    max_message_history: int = MAX_MESSAGE_HISTORY,
) -> BaseModel:
    """Run LLM with tools until it produces final structured output.

    Uses the "output as tool" pattern: the output_schema is wrapped as a
    submit_result tool. The LLM must call this tool to provide its final answer,
    ensuring Anthropic's native schema validation is applied.

    Args:
        llm: ChatAnthropic instance (unbounded)
        tools: List of LangChain tools to make available
        messages: Initial message list [SystemMessage, HumanMessage, ...]
        output_schema: Pydantic model for final output
        max_tool_calls: Maximum number of tool calls allowed (excluding submit_result)
        max_total_chars: Maximum characters from tool results
        max_message_history: Maximum messages to retain (prevents unbounded growth)

    Returns:
        Parsed output matching output_schema
    """
    budget = AgentBudget(
        max_tool_calls=max_tool_calls,
        max_total_chars=max_total_chars,
    )

    # Create submit_result tool from output schema
    submit_tool = create_submit_tool(output_schema)

    # Build tool lookup (include submit_result)
    all_tools = tools + [submit_tool]
    tools_by_name = {tool.name: tool for tool in all_tools}

    # Bind all tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # Copy messages to avoid mutating input
    working_messages = list(messages)

    # Add instruction to use submit_result
    if working_messages and isinstance(working_messages[-1], HumanMessage):
        # Append to existing user message
        original_content = working_messages[-1].content
        working_messages[-1] = HumanMessage(
            content=f"{original_content}\n\nWhen you have completed your analysis, you MUST call the submit_result tool to provide your final output."
        )

    while True:
        # Prune message history to prevent unbounded growth
        working_messages = prune_message_history(
            working_messages,
            max_history=max_message_history,
        )

        # Pre-flight token check
        try:
            preflight_token_check(working_messages, has_tools=True)
        except TokenBudgetExceeded as e:
            logger.warning(f"Pre-flight token check failed: {e}")
            # Force immediate submission
            working_messages.append(
                HumanMessage(
                    content=(
                        "CRITICAL: Context is approaching token limits. "
                        "You MUST call submit_result NOW with your best current analysis. "
                        "Do not make any more tool calls."
                    )
                )
            )

        # Invoke LLM
        response: AIMessage = await llm_with_tools.ainvoke(working_messages)
        working_messages.append(response)

        # Check for tool calls
        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            # No tool calls - remind LLM to use submit_result
            logger.warning("Agent returned without tool calls, prompting for submit_result")
            working_messages.append(
                HumanMessage(
                    content="You must call the submit_result tool to provide your final output."
                )
            )

            if not budget.can_continue():
                # Force structured output as last resort
                break
            continue

        # Process tool calls
        for tool_call in tool_calls:
            tool_id = tool_call.get("id", "")
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            # Check if this is the submit_result call
            if tool_name == "submit_result":
                logger.info(f"Agent submitted result. {budget.get_status()}")
                # Parse and validate the args as our output schema
                try:
                    result = parse_submit_result(tool_args, output_schema)
                    return result
                except Exception as e:
                    logger.error(f"Failed to validate submit_result args: {e}")
                    # Add error and continue
                    working_messages.append(
                        ToolMessage(
                            content=json.dumps({"error": f"Invalid output: {e}. Please try again."}),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                    continue

            # Execute regular tool
            logger.info(f"Executing tool: {tool_name}")
            result = await execute_tool(tool_call, tools_by_name)

            # Record and check budget
            budget.record_tool_call(len(result))

            # Add tool result to messages
            working_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

            logger.debug(f"Tool {tool_name} returned {len(result)} chars")

        # Check budget after processing all tool calls
        if not budget.can_continue():
            # Determine which limit was hit
            if budget.is_char_budget_exceeded():
                reason = f"Character budget exceeded ({budget.chars_retrieved:,}/{budget.max_total_chars:,} chars)"
            else:
                reason = f"Tool call limit reached ({budget.tool_calls_made}/{budget.max_tool_calls} calls)"

            logger.info(f"Budget exceeded: {reason}. {budget.get_status()}")
            working_messages.append(
                HumanMessage(
                    content=(
                        f"Budget limit reached: {reason}. "
                        "You must call submit_result now to provide your final output. "
                        "Do not make any more tool calls except submit_result."
                    )
                )
            )

    # Fallback: force structured output directly
    logger.info(f"Forcing final output via structured_output. {budget.get_status()}")

    MAX_RETRIES = 2
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            structured_llm = llm.with_structured_output(output_schema, method="json_schema")
            final_response = await structured_llm.ainvoke(working_messages)
            return final_response
        except Exception as e:
            last_error = e
            logger.warning(f"Structured output attempt {attempt + 1} failed: {e}")
            continue

    # All retries failed - raise with context
    raise ValueError(
        f"Failed to parse {output_schema.__name__} after {MAX_RETRIES} attempts: {last_error}"
    )


async def run_tool_agent_simple(
    llm,
    tools: list[BaseTool],
    system_prompt: str,
    user_prompt: str,
    output_schema: Type[BaseModel],
    max_tool_calls: int = 12,
) -> BaseModel:
    """Convenience wrapper for run_tool_agent with simple prompts.

    Args:
        llm: ChatAnthropic instance
        tools: List of tools to make available
        system_prompt: System message content
        user_prompt: User message content
        output_schema: Expected output schema
        max_tool_calls: Tool call budget

    Returns:
        Parsed output matching output_schema
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    return await run_tool_agent(
        llm=llm,
        tools=tools,
        messages=messages,
        output_schema=output_schema,
        max_tool_calls=max_tool_calls,
    )


__all__ = ["run_tool_agent", "run_tool_agent_simple"]
