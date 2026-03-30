"""MCP server exposing paper corpus tools for CLI-based LLM invocations."""

__all__ = ["main"]


def main():
    """Entry point for the paper tools MCP server."""
    import asyncio

    from .server import main as _main

    asyncio.run(_main())
