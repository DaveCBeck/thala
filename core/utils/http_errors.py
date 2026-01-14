"""HTTP error handling utilities."""

import logging
from typing import Any, Type

import httpx

logger = logging.getLogger(__name__)


async def safe_http_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    error_class: Type[Exception],
    **kwargs: Any,
) -> httpx.Response:
    """
    Make HTTP request with consistent error handling.

    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (GET, POST, etc.)
        path: Request path
        error_class: Exception class to raise on error
        **kwargs: Additional arguments for request

    Returns:
        Response object

    Raises:
        error_class: On HTTP or connection errors
    """
    try:
        response = await client.request(method, path, **kwargs)
        response.raise_for_status()
        return response
    except httpx.ConnectError as e:
        logger.error(f"Connection failed to {client.base_url}{path}: {e}")
        raise error_class(f"Connection failed: {e}")
    except httpx.TimeoutException as e:
        logger.error(f"Request timeout for {client.base_url}{path}: {e}")
        raise error_class(f"Request timeout: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} error for {client.base_url}{path}")
        raise error_class(f"HTTP {e.response.status_code}: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error for {client.base_url}{path}: {e}")
        raise error_class(f"Request failed: {e}")
