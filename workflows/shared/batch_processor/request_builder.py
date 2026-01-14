"""Request building logic for batch processing."""

from typing import Any

from .models import BatchRequest, sanitize_custom_id


class RequestBuilder:
    """Builds batch requests in Anthropic API format."""

    def __init__(self):
        self._id_mapping: dict[str, str] = {}  # sanitized -> original

    def build_batch_requests(self, requests: list[BatchRequest]) -> list[dict]:
        """Convert pending requests to API format.

        Automatically sanitizes custom_ids and stores mapping for result lookup.
        """
        batch_requests = []
        for req in requests:
            sanitized_id = sanitize_custom_id(req.custom_id)
            self._id_mapping[sanitized_id] = req.custom_id

            params: dict[str, Any] = {
                "model": req.model.value,
                "max_tokens": req.max_tokens,
                "messages": [{"role": "user", "content": req.prompt}],
            }

            if req.system:
                params["system"] = req.system

            if req.thinking_budget:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": req.thinking_budget,
                }
                # Ensure max_tokens > thinking_budget
                if req.max_tokens <= req.thinking_budget:
                    params["max_tokens"] = req.thinking_budget + 4096

            if req.tools:
                params["tools"] = req.tools
            if req.tool_choice:
                params["tool_choice"] = req.tool_choice

            batch_requests.append(
                {
                    "custom_id": sanitized_id,
                    "params": params,
                }
            )

        return batch_requests

    def get_original_id(self, sanitized_id: str) -> str:
        """Map sanitized custom_id back to original."""
        return self._id_mapping.get(sanitized_id, sanitized_id)

    def clear(self) -> None:
        """Clear ID mapping."""
        self._id_mapping.clear()
