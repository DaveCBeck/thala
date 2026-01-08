"""Agent runner for tool-enabled LLM interactions.

Provides a simple agent loop that:
1. Invokes LLM with tools + output schema as a tool
2. Executes any tool calls
3. When "submit_result" tool is called, extracts and returns the structured output
4. Continues until submit_result or budget exceeded
"""

import json
import logging
from typing import Any, Type

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentBudget(BaseModel):
    """Budget tracking for agent execution."""

    max_tool_calls: int = 10
    max_total_chars: int = 50000
    tool_calls_made: int = 0
    chars_retrieved: int = 0

    def can_continue(self) -> bool:
        """Check if budget allows more tool calls."""
        return self.tool_calls_made < self.max_tool_calls

    def record_tool_call(self, result_chars: int) -> None:
        """Record a tool call and its result size."""
        self.tool_calls_made += 1
        self.chars_retrieved += result_chars

    def get_status(self) -> str:
        """Get budget status string."""
        return (
            f"[Budget: {self.tool_calls_made}/{self.max_tool_calls} calls, "
            f"{self.chars_retrieved}/{self.max_total_chars} chars]"
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


def create_submit_tool(output_schema: Type[BaseModel]) -> StructuredTool:
    """Create a submit_result tool from the output schema.

    This tool is used by the agent to provide its final structured output.
    Using a tool ensures Anthropic's schema validation is applied.
    """
    def submit_result(**kwargs) -> str:
        """Submit the final result. Call this when you have completed your analysis."""
        return json.dumps({"status": "submitted"})

    return StructuredTool.from_function(
        func=submit_result,
        name="submit_result",
        description=(
            "Submit your final result. Call this tool when you have completed your analysis "
            "and are ready to provide your structured output. You MUST call this tool to finish."
        ),
        args_schema=output_schema,
    )


async def run_tool_agent(
    llm,
    tools: list[BaseTool],
    messages: list,
    output_schema: Type[BaseModel],
    max_tool_calls: int = 10,
    max_total_chars: int = 50000,
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
                    result = output_schema.model_validate(tool_args)
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
            logger.info(f"Budget exceeded. {budget.get_status()}")
            working_messages.append(
                HumanMessage(
                    content=(
                        "Budget limit reached. You must call submit_result now to provide your final output. "
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
    max_tool_calls: int = 10,
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
