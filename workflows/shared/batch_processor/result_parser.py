"""Result parsing logic for batch processing."""

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

from .models import BatchResult


class ResultParser:
    """Parses batch results from Anthropic API."""

    def __init__(self, id_mapper: callable):
        """Initialize with a function to map sanitized IDs to original IDs."""
        self._get_original_id = id_mapper

    async def fetch_results(self, results_url: str) -> dict[str, BatchResult]:
        """Fetch and parse batch results from the results URL.

        Results are keyed by original (unsanitized) custom_id for caller convenience.
        """
        results: dict[str, BatchResult] = {}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                results_url,
                headers={
                    "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                    "anthropic-version": "2023-06-01",
                },
            )
            response.raise_for_status()

            # Results are JSONL format (one JSON object per line)
            for line in response.text.strip().split("\n"):
                if not line:
                    continue

                result_data = json.loads(line)
                sanitized_id = result_data["custom_id"]
                original_id = self._get_original_id(sanitized_id)
                result = result_data["result"]

                if result["type"] == "succeeded":
                    message = result["message"]
                    content = ""
                    thinking = None

                    # Extract content from message
                    for block in message.get("content", []):
                        if block.get("type") == "text":
                            content = block.get("text", "")
                        elif block.get("type") == "thinking":
                            thinking = block.get("thinking", "")
                        elif block.get("type") == "tool_use":
                            # Tool input is already valid JSON - serialize it
                            tool_input = block.get("input", {})
                            # Unwrap $output wrapper if present (Anthropic batch API wrapping)
                            if isinstance(tool_input, dict) and "$output" in tool_input and len(tool_input) == 1:
                                tool_input = tool_input["$output"]
                            logger.debug(f"[DIAG] tool_use block input keys: {list(tool_input.keys()) if isinstance(tool_input, dict) else type(tool_input)}")
                            content = json.dumps(tool_input)

                    results[original_id] = BatchResult(
                        custom_id=original_id,
                        success=True,
                        content=content,
                        thinking=thinking,
                        usage=message.get("usage"),
                    )
                elif result["type"] == "errored":
                    error = result.get("error", {})
                    # Handle nested error structure from Anthropic batch API
                    # Structure: {"type": "error", "error": {"type": "...", "message": "..."}}
                    if error.get("type") == "error" and "error" in error:
                        inner_error = error["error"]
                        error_msg = f"{inner_error.get('type', 'unknown')}: {inner_error.get('message', 'Unknown error')}"
                    else:
                        error_msg = f"{error.get('type', 'unknown')}: {error.get('message', 'Unknown error')}"
                    results[original_id] = BatchResult(
                        custom_id=original_id,
                        success=False,
                        error=error_msg,
                    )
                elif result["type"] in ("canceled", "expired"):
                    results[original_id] = BatchResult(
                        custom_id=original_id,
                        success=False,
                        error=f"Request {result['type']}",
                    )

        return results
