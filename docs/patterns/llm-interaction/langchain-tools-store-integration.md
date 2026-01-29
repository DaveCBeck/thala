---
name: langchain-tools-store-integration
title: "LangChain Tools for Store Integration"
date: 2025-12-17
category: llm-interaction
applicability:
  - "When building LangChain agents that need access to internal stores"
  - "When LLM should decide when to search or expand context"
  - "When multiple heterogeneous stores need unified access patterns"
components: [mcp_tool, elasticsearch, chroma, zotero, async_task]
complexity: moderate
verified_in_production: false
related_solutions: []
tags: [langchain, tools, stores, integration, agents, lazy-initialization]
---

# LangChain Tools for Store Integration

## Intent

Expose internal store functionality (Elasticsearch, ChromaDB, Zotero) to LangChain agents through well-defined tools with Pydantic schemas and lazy initialization.

## Motivation

When building agentic workflows where the LLM decides when to search memory or expand context, you need tools that:
- Have clear input/output schemas for the LLM to understand
- Handle multiple heterogeneous stores with different APIs
- Initialize expensive connections only when needed
- Provide resilient error handling when individual stores fail

This pattern creates a clean integration layer between internal stores and LangChain's agent framework.

## Applicability

Use this pattern when:
- Building LangChain agents with multi-store dependencies
- Tools need environment-based configuration (centralized)
- Store instances are expensive to initialize (network I/O, authentication)
- You want semantic search capabilities integrated with external tool frameworks

Do NOT use this pattern when:
- Tools are simple without external dependencies
- Store configuration is tool-specific rather than global
- You need synchronous-only execution (this pattern is async-first)

## Structure

```
langchain_tools/
├── __init__.py          # Curated public API exports
├── base.py              # StoreManager + get_store_manager()
├── search_memory.py     # Cross-store semantic search tool
└── expand_context.py    # Deep-dive retrieval tool
```

The StoreManager centralizes store access with lazy initialization:

```
                     ┌─────────────────────┐
                     │   StoreManager      │
                     │ (lazy init, env)    │
                     └─────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ElasticsearchStores│  │  ChromaStore    │  │   ZoteroStore   │
│ (coherence, store) │  │ (top_of_mind)   │  │  (metadata)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Implementation

### Step 1: Create StoreManager with Lazy Initialization

The StoreManager holds configuration and creates store instances on first access.

```python
"""langchain_tools/base.py"""
import os
from typing import Optional

from core.stores.elasticsearch import ElasticsearchStores
from core.stores.chroma import ChromaStore
from core.stores.zotero import ZoteroStore
from core.embedding import EmbeddingService


class StoreManager:
    """
    Manages store connections for LangChain tools.
    Uses lazy initialization - stores are created on first access.
    """

    _es_stores: Optional[ElasticsearchStores] = None
    _chroma: Optional[ChromaStore] = None
    _zotero: Optional[ZoteroStore] = None
    _embedding: Optional[EmbeddingService] = None

    def __init__(
        self,
        es_coherence_host: Optional[str] = None,
        es_forgotten_host: Optional[str] = None,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
    ):
        # Configuration captured at init time, stores created on demand
        self._es_coherence_host = es_coherence_host or os.environ.get(
            "THALA_ES_COHERENCE_HOST", "http://localhost:9201"
        )
        self._chroma_host = chroma_host or os.environ.get(
            "THALA_CHROMA_HOST", "localhost"
        )
        # ... additional config

    @property
    def es_stores(self) -> ElasticsearchStores:
        """Get Elasticsearch stores (lazy init)."""
        if self._es_stores is None:
            self._es_stores = ElasticsearchStores(
                coherence_host=self._es_coherence_host,
            )
        return self._es_stores

    @property
    def chroma(self) -> ChromaStore:
        """Get ChromaDB store (lazy init)."""
        if self._chroma is None:
            self._chroma = ChromaStore(host=self._chroma_host)
        return self._chroma

    async def close(self):
        """Close all store connections."""
        if self._es_stores is not None:
            await self._es_stores.close()


# Singleton for convenience
_default_manager: Optional[StoreManager] = None


def get_store_manager() -> StoreManager:
    """Get the default StoreManager instance (creates if needed)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StoreManager()
    return _default_manager
```

### Step 2: Create Tool with @tool Decorator

Use LangChain's `@tool` decorator for simple function-based tools.

```python
"""langchain_tools/search_memory.py"""
from typing import Literal, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
from .base import get_store_manager


class MemorySearchResult(BaseModel):
    """Individual search result from memory."""
    id: str
    source_store: str
    content: str
    score: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class SearchMemoryOutput(BaseModel):
    """Output schema for search_memory tool."""
    query: str
    total_results: int
    results: list[MemorySearchResult]


@tool
async def search_memory(
    query: str,
    limit: int = 10,
    stores: Optional[list[Literal["top_of_mind", "coherence", "store"]]] = None,
) -> dict:
    """Search across the memory system for relevant information.

    Use this when you need to:
    - Recall something you might have learned before
    - Find beliefs, preferences, or identity information
    - Look up stored knowledge on a topic

    Args:
        query: What to search for in memory
        limit: Max results per store (default 10)
        stores: Specific stores to search (defaults to all)
    """
    store_manager = get_store_manager()
    results: list[MemorySearchResult] = []

    if stores is None:
        stores = ["top_of_mind", "coherence", "store"]

    # Search each store with graceful error handling
    if "top_of_mind" in stores:
        try:
            query_embedding = await store_manager.embedding.embed(query)
            chroma_results = await store_manager.chroma.search(
                query_embedding=query_embedding,
                n_results=limit,
            )
            for r in chroma_results:
                results.append(MemorySearchResult(
                    id=str(r["id"]),
                    source_store="top_of_mind",
                    content=r["document"] or "",
                    score=1 - r["distance"],
                    metadata=r["metadata"] or {},
                ))
        except Exception as e:
            logger.warning(f"top_of_mind search failed: {e}")
            # Continue with other stores

    # ... similar for other stores

    return SearchMemoryOutput(
        query=query,
        total_results=len(results),
        results=results,
    ).model_dump(mode="json")
```

### Step 3: Export Clean Public API

Curate what consumers can import.

```python
"""langchain_tools/__init__.py"""
from .base import StoreManager, get_store_manager
from .search_memory import search_memory, SearchMemoryOutput, MemorySearchResult
from .expand_context import expand_context, ExpandedContext

__all__ = [
    "StoreManager",
    "get_store_manager",
    "search_memory",
    "SearchMemoryOutput",
    "MemorySearchResult",
    "expand_context",
    "ExpandedContext",
]
```

## Complete Example

```python
"""
Example: Using LangChain tools in an agent.
"""
from langchain_tools import search_memory, expand_context

# Tools automatically use the default StoreManager singleton
tools = [search_memory, expand_context]

# Usage in agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Agent decides when to call search_memory or expand_context
result = await executor.ainvoke({"input": "What do I believe about AI safety?"})
```

## Consequences

### Benefits

- **Resource efficiency**: Stores created only when tools are actually used
- **Centralized configuration**: All store config via environment variables in one place
- **Resilient execution**: One store failure doesn't break the entire tool
- **Clean API**: Pydantic schemas make tool inputs/outputs clear to LLM
- **Easy testing**: Inject mock StoreManager for unit tests

### Trade-offs

- **Global state**: Singleton pattern means shared state across tools
- **Async-only**: Must use `ainvoke()`, not `invoke()` for synchronous calls
- **Extra abstraction**: Additional layer between LLM and raw store APIs
- **First-call latency**: Lazy initialization means first tool call may be slower

### Alternatives

- **Direct DI**: Pass store instances to each tool - more verbose but no global state
- **Tool factory**: Create tool instances with per-tool config - flexible but can't share connections
- **MCP tools**: Use MCP server directly instead of LangChain tools - different integration pattern

## Related Patterns

- [MCP Server Store Exposure](../stores/mcp-server-store-exposure.md): Exposing stores via MCP protocol
- [BaseTool to Decorator Migration](./langchain-basetool-to-decorator-migration.md): How to migrate from legacy BaseTool classes
- [LangChain Tools Integration (Legacy)](./langchain-tools-integration.md): Original BaseTool pattern documentation

## Known Uses in Thala

- `langchain_tools/search_memory.py`: Cross-store semantic search
- `langchain_tools/expand_context.py`: Deep-dive retrieval by UUID, Zotero key, or content snippet
- `langchain_tools/base.py`: StoreManager singleton pattern

## References

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
- [Pydantic BaseModel](https://docs.pydantic.dev/latest/concepts/models/)
