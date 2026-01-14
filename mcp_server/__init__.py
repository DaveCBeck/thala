"""MCP server for Thala knowledge stores."""

__all__ = ["main"]


def main():
    """Entry point for the MCP server."""
    import asyncio
    from .server import main as _main

    asyncio.run(_main())
