---
name: conditional-development-tracing
title: "Conditional Development Tracing with LangSmith"
date: 2025-12-18
category: llm-interaction
applicability:
  - "When debugging LangChain/LangGraph workflow stages and intermediate LLM responses"
  - "When needing observability tools in development with zero production overhead"
  - "When multiple entry points require consistent tracing configuration"
components: [llm_call, workflow_graph, configuration]
complexity: simple
verified_in_production: true
related_solutions: []
tags: [langsmith, tracing, development, observability, configuration, environment]
---

# Conditional Development Tracing with LangSmith

## Intent

Enable LangSmith tracing for LLM workflow debugging in development mode while ensuring zero overhead in production through environment-based conditional configuration.

## Motivation

LangChain and LangGraph workflows can be opaque - when something goes wrong, it's difficult to see intermediate LLM responses, token usage, and state transitions. LangSmith provides excellent visibility, but has a critical quirk: it reads environment variables at **module import time**, not at runtime.

This creates a challenge for multi-entry-point applications (CLI tools, MCP servers, workflow entry points) where:

1. Tracing must be configured **before** any LangChain imports
2. Production deployments must have zero tracing overhead
3. The same configuration logic shouldn't be duplicated across files
4. Developers need a single switch to enable/disable tracing

This pattern solves these problems with a centralized, idempotent configuration function called at each entry point.

## Applicability

Use this pattern when:
- You have LangChain/LangGraph workflows that need debugging
- Your application has multiple entry points (MCP server, CLI, HTTP endpoints)
- You need to inspect intermediate LLM responses during development
- Production must have zero tracing overhead

Do NOT use this pattern when:
- You have a single-file script (just set environment variables directly)
- You need dynamic runtime toggling of tracing (requires different approach)
- Tracing should always be enabled regardless of environment

## Structure

```
                    +-------------------+
                    |   .env file       |
                    |   THALA_MODE=dev  |
                    +--------+----------+
                             |
                             v
+----------------------------+----------------------------+
|                     core/config.py                      |
|  +------------------+    +------------------------+     |
|  | is_dev_mode()    |--->| configure_langsmith()  |     |
|  +------------------+    +------------------------+     |
|                                    |                    |
|                                    v                    |
|                    +-----------------------------+      |
|                    | os.environ modification     |      |
|                    | - LANGSMITH_TRACING         |      |
|                    | - LANGSMITH_PROJECT         |      |
|                    +-----------------------------+      |
+----------------------------+----------------------------+
                             |
                             | (must happen BEFORE)
                             v
              +-----------------------------+
              | LangChain Module Imports    |
              | - langchain_anthropic       |
              | - langchain_core            |
              | - langgraph                 |
              +-----------------------------+
```

## Implementation

### Step 1: Create the Configuration Module

Create a centralized configuration module that handles mode detection and LangSmith setup.

```python
# core/config.py
"""Thala configuration and environment setup.

This module provides centralized configuration for the Thala system,
including development mode detection and LangSmith tracing setup.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def is_dev_mode() -> bool:
    """Check if running in development mode.

    Returns:
        True if THALA_MODE is set to 'dev', False otherwise.
    """
    return os.getenv("THALA_MODE", "prod").lower() == "dev"


def configure_langsmith() -> None:
    """Configure LangSmith tracing based on THALA_MODE.

    When THALA_MODE=dev:
        - Enables LangSmith tracing
        - Sets project to 'thala-dev'

    When THALA_MODE=prod (or unset):
        - Disables LangSmith tracing

    This function is idempotent and safe to call multiple times.
    """
    if is_dev_mode():
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "thala-dev")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
```

### Step 2: Apply at Every Entry Point

The critical pattern is calling `configure_langsmith()` **before** any LangChain imports.

```python
# Entry point file (e.g., mcp_server/server.py)

# Phase 1: Load environment FIRST
from dotenv import load_dotenv
load_dotenv()

# Phase 2: Configure LangSmith BEFORE any LangChain imports
from core.config import configure_langsmith
configure_langsmith()

# Phase 3: NOW import LangChain modules (they read env vars at import time)
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
```

### Step 3: Document Environment Variables

Add documentation to `.env.example` for discoverability.

```bash
# .env.example

# =============================================================================
# Development Mode & Tracing
# =============================================================================

# Mode: "dev" enables LangSmith tracing, "prod" disables it
THALA_MODE=prod

# LangSmith (only needed when THALA_MODE=dev)
# Get your API key at https://smith.langchain.com/
LANGSMITH_API_KEY=lsv2_...
```

## Complete Example

```python
"""
MCP server entry point demonstrating the conditional tracing pattern.
From: mcp_server/server.py
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

# Phase 1: Load environment variables FIRST
load_dotenv()

# Phase 2: Configure tracing BEFORE LangChain imports
from core.config import configure_langsmith

configure_langsmith()

# Phase 3: NOW safe to import LangChain modules
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# If this module uses LangChain internally, it will now respect
# the tracing configuration we just set
from workflows.document_processing.graph import process_document


async def main():
    """Run MCP server with tracing enabled in dev mode."""
    server = Server("thala")

    # ... server implementation

    async with stdio_server() as streams:
        await server.run(streams[0], streams[1])


if __name__ == "__main__":
    asyncio.run(main())
```

## Consequences

### Benefits

- **Single control point**: One environment variable (`THALA_MODE`) controls all tracing
- **Zero production overhead**: `LANGSMITH_TRACING=false` means no network calls or processing
- **Idempotent configuration**: `os.environ.setdefault()` is safe for multiple calls from multiple entry points
- **Project isolation**: Dev traces go to `thala-dev`, keeping production data clean
- **Default secure**: Production mode is the default (tracing disabled if `THALA_MODE` unset)

### Trade-offs

- **Import ordering discipline**: Developers must follow the pattern exactly at every entry point
- **Repeated boilerplate**: Same 5-line import pattern appears in multiple files
- **No dynamic toggle**: Cannot enable/disable tracing without process restart
- **Entry point fragility**: Missing the pattern in one entry point breaks observability for that code path

### Alternatives

- **Lazy configuration in LLM factory**: Configure tracing inside `get_llm()`. However, this fails for LangGraph and other modules that need tracing before calling `get_llm()`.
- **Import hook / site-packages customization**: Auto-configure before any import. Too magical and hard to debug.
- **Runtime-only tracing via callbacks**: Pass tracing callbacks explicitly. Loses LangSmith's automatic rich tracing.

## Related Patterns

- [Centralized Environment Configuration](../stores/centralized-env-config.md) - Environment variable loading pattern
- [LangSmith Tracing Infrastructure](./langsmith-tracing-infrastructure.md) - Advanced tracing with truncation and metadata (if documented)

## Known Uses in Thala

- `langchain_tools/base.py`: Entry point for LangChain tool initialization
- `mcp_server/server.py`: MCP server entry point
- `workflows/document_processing/graph.py`: Document processing workflow entry
- `workflows/shared/llm_utils.py`: Shared LLM utilities module

## References

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Tracing Configuration](https://python.langchain.com/docs/how_to/debugging/)
