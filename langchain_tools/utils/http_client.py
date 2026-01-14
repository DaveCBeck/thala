"""HTTP client utilities for LangChain tools."""

from typing import Awaitable, Callable, Optional

from httpx import AsyncClient


def create_lazy_client(
    client_var_name: str,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    headers_factory: Optional[Callable[[], dict]] = None,
    timeout: float = 30.0,
) -> tuple[Callable[[], AsyncClient], Callable[[], Awaitable[None]]]:
    """Create lazy-initialized client getter and closer functions.

    Args:
        client_var_name: Name for the client variable (used for error messages)
        base_url: Base URL for the client
        api_key_env: Environment variable name for API key (optional)
        headers_factory: Function that returns headers dict (optional)
        timeout: Request timeout in seconds (default 30.0)

    Returns:
        Tuple of (get_client, close_client) functions
    """
    import os

    _client = None

    def get_client() -> AsyncClient:
        """Get or create the lazy-initialized client."""
        nonlocal _client
        if _client is None:
            # Build headers
            headers = {}
            if headers_factory:
                headers = headers_factory()
            elif api_key_env:
                api_key = os.environ.get(api_key_env)
                if not api_key:
                    raise ValueError(f"{api_key_env} environment variable is required")
                headers["Authorization"] = f"Bearer {api_key}"
                headers["Content-Type"] = "application/json"

            # Create client
            _client = AsyncClient(
                base_url=base_url or "",
                headers=headers,
                timeout=timeout,
            )
        return _client

    async def close_client() -> None:
        """Close the client and release resources."""
        nonlocal _client
        if _client is not None:
            await _client.aclose()
            _client = None

    return get_client, close_client
