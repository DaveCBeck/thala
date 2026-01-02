"""Utilities for safe node execution."""

import functools
import logging
from typing import Callable, Any


def safe_node_execution(
    node_name: str,
    logger: logging.Logger,
    fallback_status: str = "failed",
):
    """Decorator for safe async node execution with error handling."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state: dict, *args, **kwargs) -> dict:
            try:
                return await fn(state, *args, **kwargs)
            except Exception as e:
                logger.error(f"{node_name} failed: {e}")
                return {
                    "current_status": fallback_status,
                    "errors": [{"node": node_name, "error": str(e)}],
                }
        return wrapper
    return decorator


class StateUpdater:
    """Helpers for consistent state updates."""

    @staticmethod
    def success(status: str, **kwargs) -> dict:
        return {"current_status": status, **kwargs}

    @staticmethod
    def error(node_name: str, error: Exception | str, status: str = "failed", **kwargs) -> dict:
        return {
            "current_status": status,
            "errors": [{"node": node_name, "error": str(error)}],
            **kwargs
        }
