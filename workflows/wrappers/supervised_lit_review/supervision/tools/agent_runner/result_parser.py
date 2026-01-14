"""Result parsing and formatting for agent execution."""

import json
import logging
from typing import Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


def parse_submit_result(
    tool_args: dict,
    output_schema: Type[BaseModel],
) -> BaseModel:
    """Parse and validate submit_result tool arguments.

    Args:
        tool_args: Arguments from submit_result tool call
        output_schema: Expected output schema

    Returns:
        Validated output matching schema

    Raises:
        Exception: If validation fails
    """
    return output_schema.model_validate(tool_args)


__all__ = ["create_submit_tool", "parse_submit_result"]
