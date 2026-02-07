"""Batch result fetching and validation for the LLM Broker.

Handles fetching JSONL results from Anthropic's Batch API,
including SSRF protection via URL validation.
"""

import json
import os
from typing import Any
from urllib.parse import urlparse

import httpx

# Allowed hostnames for batch results URLs (SSRF protection)
ALLOWED_RESULTS_HOSTS = frozenset({"api.anthropic.com", "batches.anthropic.com"})


def validate_results_url(url: str) -> bool:
    """Validate that a results URL points to an allowed Anthropic domain.

    Prevents SSRF attacks by ensuring we only send API credentials to
    trusted Anthropic hostnames over HTTPS.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is valid and safe to use with credentials
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in ALLOWED_RESULTS_HOSTS
    except Exception:
        return False


async def fetch_batch_results(
    results_url: str,
    id_mapping: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Fetch and parse batch results from Anthropic.

    Args:
        results_url: URL to fetch results from
        id_mapping: Mapping of sanitized custom_id -> original request_id.
            Processed entries are removed in-place to prevent unbounded growth.

    Returns:
        Dictionary of request_id -> result data

    Raises:
        ValueError: If results_url does not point to an allowed Anthropic domain
    """
    # Validate URL to prevent SSRF - only send API key to trusted Anthropic domains
    if not validate_results_url(results_url):
        raise ValueError(
            f"Invalid batch results URL: must be HTTPS to an allowed Anthropic domain "
            f"({', '.join(sorted(ALLOWED_RESULTS_HOSTS))}), got: {results_url}"
        )

    results: dict[str, dict[str, Any]] = {}
    processed_sanitized_ids: list[str] = []

    async with httpx.AsyncClient() as client:
        response = await client.get(
            results_url,
            headers={
                "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
            },
        )
        response.raise_for_status()

        # Results are JSONL format
        for line in response.text.strip().split("\n"):
            if not line:
                continue

            result_data = json.loads(line)
            sanitized_id = result_data["custom_id"]
            original_id = id_mapping.get(sanitized_id, sanitized_id)
            processed_sanitized_ids.append(sanitized_id)
            result = result_data["result"]

            if result["type"] == "succeeded":
                message = result["message"]
                content = None
                thinking = None

                for block in message.get("content", []):
                    if block.get("type") == "text":
                        content = block.get("text", "")
                    elif block.get("type") == "thinking":
                        thinking = block.get("thinking", "")
                    elif block.get("type") == "tool_use":
                        tool_input = block.get("input", {})
                        if isinstance(tool_input, dict) and "$output" in tool_input and len(tool_input) == 1:
                            tool_input = tool_input["$output"]
                        content = json.dumps(tool_input)

                results[original_id] = {
                    "success": True,
                    "content": content,
                    "thinking": thinking,
                    "usage": message.get("usage"),
                    "model": message.get("model"),
                    "stop_reason": message.get("stop_reason"),
                }

            elif result["type"] == "errored":
                error = result.get("error", {})
                if error.get("type") == "error" and "error" in error:
                    inner_error = error["error"]
                    error_msg = (
                        f"{inner_error.get('type', 'unknown')}: {inner_error.get('message', 'Unknown error')}"
                    )
                else:
                    error_msg = f"{error.get('type', 'unknown')}: {error.get('message', 'Unknown error')}"
                results[original_id] = {
                    "success": False,
                    "error": error_msg,
                }

            elif result["type"] in ("canceled", "expired"):
                results[original_id] = {
                    "success": False,
                    "error": f"Request {result['type']}",
                }

    # Cleanup processed ID mappings to prevent unbounded memory growth
    for sanitized_id in processed_sanitized_ids:
        id_mapping.pop(sanitized_id, None)

    return results
