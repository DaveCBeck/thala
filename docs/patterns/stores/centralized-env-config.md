---
name: centralized-env-config
title: "Centralized Environment Configuration with dotenv"
date: 2025-12-17
category: stores
applicability:
  - "Multi-module Python projects with shared environment variables"
  - "Projects with multiple entry points (CLI, server, library imports)"
  - "Codebases requiring external service credentials"
components: [configuration, embedding, mcp_tool]
complexity: simple
verified_in_production: true
related_solutions: []
tags: [dotenv, environment, configuration, entry-point, python-dotenv]
---

# Centralized Environment Configuration with dotenv

## Intent

Provide a single source of truth for environment variables with consistent loading at entry points across a multi-module Python application.

## Motivation

In a multi-module Python application with several entry points (MCP server, LangChain tools, embedding service), environment variable management becomes fragmented:

- Each module might assume variables are already loaded
- Users need to set variables differently depending on the entry point
- No single source of truth for required configuration
- Configuration documentation scattered across codebase

This pattern solves these issues by establishing:
1. A `.env.example` template documenting ALL environment variables
2. Consistent `load_dotenv()` placement at entry point modules
3. Self-contained modules that work regardless of invocation method

## Applicability

Use this pattern when:
- Your project has multiple entry points (CLI, server, tests, library imports)
- You need to document required environment variables for onboarding
- Multiple modules independently access `os.environ`
- External service credentials need clear documentation

Do NOT use this pattern when:
- Single-file scripts with few environment variables
- Environment is managed by orchestration (Kubernetes ConfigMaps, etc.)
- Secrets management system handles variable injection

## Structure

```
project/
├── .env.example          # Template with all variables documented
├── .env                  # Actual values (gitignored)
├── requirements.txt      # Includes python-dotenv
└── src/
    ├── entry_point_1.py  # load_dotenv() at top
    ├── entry_point_2.py  # load_dotenv() at top
    └── utils/
        └── helper.py     # NO load_dotenv() - not an entry point
```

## Implementation

### Step 1: Create .env.example Template

Organize variables into logical sections with documentation:

```bash
# Project Environment Variables
# Copy this file to .env and fill in your values

# =============================================================================
# API Keys (required for respective features)
# =============================================================================

# OpenAI - Required for embeddings (default provider)
OPENAI_API_KEY=sk-...

# Firecrawl - Required for web scraping tools
# Get one at https://firecrawl.dev
FIRECRAWL_API_KEY=fc-...

# =============================================================================
# Feature Configuration
# =============================================================================

# Provider: "openai" or "ollama"
THALA_EMBEDDING_PROVIDER=openai

# Model name (provider-specific)
# OpenAI: text-embedding-3-small, text-embedding-3-large
# Ollama: nomic-embed-text, mxbai-embed-large
THALA_EMBEDDING_MODEL=text-embedding-3-small

# =============================================================================
# Service Hosts (defaults work for local Docker setup)
# =============================================================================

# Elasticsearch
THALA_ES_COHERENCE_HOST=http://localhost:9201
THALA_ES_FORGOTTEN_HOST=http://localhost:9200

# ChromaDB
THALA_CHROMA_HOST=localhost
THALA_CHROMA_PORT=8000
```

**Section organization:**
1. **API Keys** - External service credentials with links to get them
2. **Feature Configuration** - Application settings with valid options documented
3. **Service Hosts** - Infrastructure endpoints with sensible defaults

### Step 2: Add load_dotenv() to Entry Points

Place `load_dotenv()` at the top of entry point modules, immediately after imports:

```python
# core/embedding.py - Entry point for embedding service
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()  # Immediately after imports, before any env var access


class EmbeddingService:
    def __init__(self):
        self.provider = os.environ.get("THALA_EMBEDDING_PROVIDER", "openai")
        # ...
```

```python
# mcp_server/server.py - Entry point for MCP server
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
```

### Step 3: Lazy Client Initialization

For external API clients, combine with lazy initialization:

```python
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client():
    """Get API client (lazy init)."""
    global _client
    if _client is None:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            raise ValueError(
                "API_KEY environment variable is required. "
                "Get one at https://example.com"
            )
        _client = APIClient(api_key=api_key)
    return _client
```

## Complete Example

```python
"""
langchain_tools/firecrawl.py - Complete implementation from Thala.
"""
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy singleton for API client
_firecrawl_client = None


def _get_firecrawl():
    """Get AsyncFirecrawl client (lazy init)."""
    global _firecrawl_client
    if _firecrawl_client is None:
        from firecrawl import AsyncFirecrawl

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY environment variable is required. "
                "Get one at https://firecrawl.dev"
            )
        _firecrawl_client = AsyncFirecrawl(api_key=api_key)
    return _firecrawl_client


class WebSearchOutput(BaseModel):
    """Output schema for web_search tool."""
    query: str
    total_results: int
    results: list[dict]


@tool
async def web_search(query: str, limit: int = 5) -> dict:
    """Search the web for information."""
    client = _get_firecrawl()
    response = await client.search(query, limit=limit)
    # ... process response
```

## Consequences

### Benefits

- **Single source of truth**: `.env.example` documents all configuration
- **Safe redundancy**: `load_dotenv()` is idempotent - multiple calls are safe
- **Self-documenting**: New developers can see all required variables
- **Entry point independence**: Any module can be the starting point

### Trade-offs

- **Slight import overhead**: `load_dotenv()` runs at import time
- **Multiple load calls**: Each entry point loads the file (but idempotent)
- **File system dependency**: Requires `.env` file to exist (falls back to system env)

### Alternatives

- **pydantic-settings**: For more complex validation and type coercion
- **dynaconf**: For multi-environment configuration management
- **Environment injection**: Let orchestration (Docker, K8s) handle variables

## Related Patterns

- [MCP Server Store Exposure](./mcp-server-store-exposure.md) - Uses this pattern for embedding configuration
- [LangChain Tools Integration](../llm-interaction/langchain-tools-integration.md) - StoreManager env var pattern
- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - Extends this pattern for mode-based LangSmith configuration

## Known Uses in Thala

- `.env.example`: Template with all THALA_* variables documented
- `core/embedding.py`: Loads env for embedding provider selection
- `langchain_tools/base.py`: Loads env for store manager initialization
- `langchain_tools/firecrawl.py`: Loads env for Firecrawl API key
- `mcp_server/server.py`: Loads env for MCP server startup

## References

- [python-dotenv documentation](https://pypi.org/project/python-dotenv/)
- [12-Factor App: Config](https://12factor.net/config)
