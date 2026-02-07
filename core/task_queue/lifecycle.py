"""Supervisor lifecycle helpers."""

import logging

from core.llm_broker import get_broker, is_broker_enabled
from core.utils.async_http_client import cleanup_all_clients

logger = logging.getLogger(__name__)


async def cleanup_supervisor_resources() -> None:
    """Clean up shared resources owned by the supervisor (broker, HTTP clients).

    Called from the finally block of both parallel.py and queue_loop.py.
    """
    if is_broker_enabled():
        try:
            await get_broker().stop()
        except Exception:
            logger.exception("Error stopping broker")

    await cleanup_all_clients()
