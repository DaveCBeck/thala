"""Base class for async context managers."""

from typing import Optional


class AsyncContextManager:
    """Base class for async context managers with close() method."""

    async def close(self) -> None:
        """Close resources. Override in subclass."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
