"""
Graceful shutdown coordination for task queue.

Provides asyncio-native signal handling that:
- Uses loop.add_signal_handler() for proper async integration
- Allows immediate interruption of long sleeps via asyncio.Event
- Coordinates cleanup before exit

Usage:
    coordinator = ShutdownCoordinator()
    coordinator.install_signal_handlers()

    while not coordinator.shutdown_requested:
        # Do work
        if await coordinator.wait_or_shutdown(300):  # 5 min timeout
            break  # Shutdown requested, exit loop

    # Cleanup happens in finally block
"""

import asyncio
import logging
import signal
from typing import Optional

logger = logging.getLogger(__name__)


class ShutdownCoordinator:
    """Coordinate graceful shutdown across async workflows.

    Uses asyncio.Event for immediate interruption of long sleeps when
    shutdown is requested via SIGINT/SIGTERM.
    """

    def __init__(self) -> None:
        """Initialize shutdown coordinator."""
        self._shutdown_event = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._handlers_installed = False

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def request_shutdown(self) -> None:
        """Request graceful shutdown.

        Sets the shutdown event, which immediately wakes any coroutines
        waiting in wait_or_shutdown().
        """
        if not self._shutdown_event.is_set():
            logger.info("Shutdown signal received")
            self._shutdown_event.set()

    async def wait_or_shutdown(self, timeout: float) -> bool:
        """Wait for timeout or shutdown signal.

        Use this instead of asyncio.sleep() to allow immediate response
        to shutdown signals.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if shutdown was requested, False if timeout expired
        """
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            return True  # Shutdown requested
        except asyncio.TimeoutError:
            return False  # Normal timeout, continue work

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers.

        Must be called from within an async context (running event loop).
        Uses loop.add_signal_handler() for proper asyncio integration.

        Note: Only works on Unix systems. On Windows, signals are handled
        differently and this will be a no-op.
        """
        if self._handlers_installed:
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("Cannot install signal handlers: no running event loop")
            return

        try:
            self._loop.add_signal_handler(signal.SIGINT, self.request_shutdown)
            self._loop.add_signal_handler(signal.SIGTERM, self.request_shutdown)
            self._handlers_installed = True
            logger.debug("Signal handlers installed (SIGINT, SIGTERM)")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning(
                "Signal handlers not supported on this platform. "
                "Graceful shutdown via Ctrl+C may not work properly."
            )

    def remove_signal_handlers(self) -> None:
        """Remove signal handlers.

        Called during cleanup to restore default signal handling.
        """
        if not self._handlers_installed or not self._loop:
            return

        try:
            self._loop.remove_signal_handler(signal.SIGINT)
            self._loop.remove_signal_handler(signal.SIGTERM)
            self._handlers_installed = False
            logger.debug("Signal handlers removed")
        except (NotImplementedError, ValueError):
            # Windows or handlers already removed
            pass


# Global coordinator instance for use across the application
_coordinator: Optional[ShutdownCoordinator] = None


def get_shutdown_coordinator() -> ShutdownCoordinator:
    """Get or create the global shutdown coordinator.

    Returns:
        The global ShutdownCoordinator instance
    """
    global _coordinator
    if _coordinator is None:
        _coordinator = ShutdownCoordinator()
    return _coordinator


def reset_shutdown_coordinator() -> None:
    """Reset the global shutdown coordinator.

    Useful for testing or when restarting the queue loop.
    """
    global _coordinator
    if _coordinator is not None:
        _coordinator.remove_signal_handlers()
    _coordinator = None
