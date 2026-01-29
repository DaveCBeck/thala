---
name: langchain-tools-integration
title: "LangChain Tools Integration with Lazy-Loaded Store Manager (Legacy BaseTool Pattern)"
date: 2025-01-28
category: llm-interaction
applicability:
  - "Building LangChain agents that need access to multi-store data systems"
  - "Creating async-only tools with complex initialization dependencies"
  - "Designing tools that aggregate results from multiple heterogeneous sources"
  - "Implementing environment-driven configuration for tool dependencies"
components: [langchain_tools, core.stores, core.embedding]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [langchain, tools, async, stores, design-pattern, legacy, basetool]
---

# LangChain Tools Integration with Lazy-Loaded Store Manager

> **Note**: This pattern documents the **legacy BaseTool class inheritance pattern** used prior to LangChain 1.x.
> For new tools, prefer the [@tool decorator pattern](./langchain-tools-store-integration.md).
> For migrating existing BaseTool classes, see [BaseTool to Decorator Migration](./langchain-basetool-to-decorator-migration.md).

## Intent

Provide a clean, reusable pattern for integrating LangChain tools with complex, environment-configurable store dependencies using lazy initialization and async-first design.

## Motivation

LangChain tools need access to external systems (Elasticsearch, ChromaDB, Zotero), but directly initializing these connections in tool classes creates tight coupling, makes testing difficult, and wastes resources by initializing unused stores. The StoreManager pattern solves this by:

1. **Centralizing configuration** - Single source of truth for all store connections, driven by environment variables
2. **Lazy initialization** - Stores only created when first accessed (efficient resource usage)
3. **Singleton access** - Tools share the same store instances without explicit dependency injection
4. **Async-first design** - All tool operations are async-only, avoiding the synchronous/async mixing problem
5. **Clean separation** - Base utilities in `base.py`, tool implementations in separate modules

This pattern emerged from the need to give LLM agents the ability to search memory and expand context dynamically, where the model decides when to invoke tools rather than having tools called directly by application code.

## Applicability

Use this pattern when:
- Building LangChain agents with multiple data source dependencies
- Tools need environment-based configuration that should be centralized
- Store instances are expensive to initialize (network calls, authentication)
- Some stores might not be used by all tools (lazy loading saves resources)
- You want to support both injected and default store managers (flexibility)
- Tools need async-only execution model

Do NOT use this pattern when:
- Tools are simple and don't need external dependencies
- You're building synchronous-only tools (use sync wrappers instead)
- Store configuration is tool-specific rather than global
- You need different store instances per tool (use DI instead)

## Structure

```
langchain_tools/
├── __init__.py          # Public API exports
├── base.py              # StoreManager (lazy init, config)
├── search_memory.py     # SearchMemoryTool (cross-store semantic search)
└── expand_context.py    # ExpandContextTool (deep-dive retrieval)

core/
├── embedding.py         # Shared embedding service (moved from mcp_server/)
└── stores/
    ├── elasticsearch.py # ElasticsearchStores
    ├── chroma.py       # ChromaStore
    └── zotero.py       # ZoteroStore
```

## Implementation

### Step 1: Build the StoreManager with Lazy Initialization

The StoreManager centralizes configuration and provides lazy initialization:

```python
# langchain_tools/base.py

class StoreManager:
    """
    Manages store connections for LangChain tools.

    Uses lazy initialization - stores are created on first access.
    Configurable via environment variables (same as MCP server).
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
        zotero_host: Optional[str] = None,
        zotero_port: Optional[int] = None,
    ):
        """Initialize with optional custom hosts; falls back to env vars."""
        self._es_coherence_host = es_coherence_host or os.environ.get(
            "THALA_ES_COHERENCE_HOST", "http://localhost:9201"
        )
        self._es_forgotten_host = es_forgotten_host or os.environ.get(
            "THALA_ES_FORGOTTEN_HOST", "http://localhost:9200"
        )
        self._chroma_host = chroma_host or os.environ.get(
            "THALA_CHROMA_HOST", "localhost"
        )
        self._chroma_port = chroma_port or int(os.environ.get(
            "THALA_CHROMA_PORT", "8000"
        ))
        # ... more configuration parameters
```

**Key design decisions:**
- Configuration captured at init time, not at access time (immutable after creation)
- Private class variables (`_es_stores`, etc.) store cached instances
- Public properties use `@property` decorator for lazy initialization
- Explicit `close()` method for cleanup (important for resource management)

### Step 2: Implement Lazy-Loaded Properties

Each store gets a property that creates it on first access:

```python
    @property
    def es_stores(self) -> ElasticsearchStores:
        """Get Elasticsearch stores (lazy init)."""
        if self._es_stores is None:
            self._es_stores = ElasticsearchStores(
                coherence_host=self._es_coherence_host,
                forgotten_host=self._es_forgotten_host,
            )
        return self._es_stores

    @property
    def embedding(self) -> EmbeddingService:
        """Get embedding service (lazy init)."""
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    async def close(self):
        """Close all store connections."""
        if self._es_stores is not None:
            await self._es_stores.close()
        if self._embedding is not None:
            await self._embedding.close()
        # ... close other stores
```

**Advantages:**
- First access creates the store (overhead only when needed)
- Subsequent accesses return cached instance (efficient)
- Clear `close()` pattern for cleanup in async context managers
- Dependencies between stores respected (e.g., ChromaStore gets es_stores for history tracking)

### Step 3: Provide Singleton Access

Make the default instance globally available without forcing direct imports:

```python
# Singleton instance for convenience
_default_manager: Optional[StoreManager] = None

def get_store_manager() -> StoreManager:
    """Get the default StoreManager instance (creates if needed)."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StoreManager()
    return _default_manager
```

**Why this approach:**
- Tools can use `get_store_manager()` without import coupling
- Tests can inject a custom StoreManager via Pydantic Field default_factory
- No global mutable state issues (singleton created lazily, only once)

### Step 4: Create the LangChain Tool Class

Use the StoreManager in a BaseTool subclass with async-only execution:

```python
# langchain_tools/search_memory.py

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SearchMemoryInput(BaseModel):
    """Input schema for search_memory tool."""
    query: str = Field(description="What to search for in memory")
    limit: int = Field(default=10, ge=1, le=50, description="Max results per store")
    stores: Optional[list[Literal["top_of_mind", "coherence", "who_i_was", "store"]]] = Field(
        default=None,
        description="Specific stores to search. Defaults to all except who_i_was."
    )

class SearchMemoryOutput(BaseModel):
    """Output schema for search_memory tool."""
    query: str
    total_results: int
    results: list[MemorySearchResult]

class SearchMemoryTool(BaseTool):
    """Search across the memory system for relevant information."""

    name: str = "search_memory"
    description: str = "Search across the memory system..."
    args_schema: type[BaseModel] = SearchMemoryInput

    # Inject StoreManager with lazy default
    store_manager: StoreManager = Field(default_factory=get_store_manager)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> dict[str, Any]:
        """Sync wrapper - raises error, use async version."""
        raise NotImplementedError("SearchMemoryTool only supports async. Use ainvoke().")

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        stores: Optional[list[str]] = None,
        include_historical: bool = False,
    ) -> dict[str, Any]:
        """Async implementation - this is where the real work happens."""
        results: list[MemorySearchResult] = []

        # Default stores if not specified
        if stores is None:
            stores = ["top_of_mind", "coherence", "store"]

        # 1. Semantic search in top_of_mind (vector similarity)
        if "top_of_mind" in stores:
            try:
                query_embedding = await self.store_manager.embedding.embed(query)
                chroma_results = await self.store_manager.chroma.search(
                    query_embedding=query_embedding,
                    n_results=limit,
                )
                # Process results...
            except Exception as e:
                logger.warning(f"top_of_mind search failed: {e}")

        # 2. Text search in coherence, store, who_i_was (similar pattern)
        # ...

        return {"query": query, "total_results": len(results), "results": results}
```

**Key patterns:**
- `_run()` raises NotImplementedError (tools are async-only)
- `_arun()` is the actual implementation (async)
- `store_manager` is a Field with `default_factory=get_store_manager` (for injection)
- `Config.arbitrary_types_allowed = True` (tells Pydantic we're using non-standard types)
- Exception handling per-store (failure in one store doesn't break others)

### Step 5: Cross-Store Aggregation

Combine results from multiple heterogeneous sources:

```python
        # 4. Collect results from different store types

        # top_of_mind returns similarity scores
        if "top_of_mind" in stores:
            for r in chroma_results:
                results.append(MemorySearchResult(
                    id=str(r["id"]),
                    source_store="top_of_mind",
                    content=r["document"] or "",
                    score=1 - r["distance"],  # Convert distance to similarity
                    zotero_key=r["metadata"].get("zotero_key"),
                ))

        # coherence returns Elasticsearch matches
        if "coherence" in stores:
            for r in coherence_results:
                results.append(MemorySearchResult(
                    id=str(r.id),
                    source_store="coherence",
                    content=r.content,
                    score=None,  # ES scores not directly comparable
                    metadata={"category": r.category, "confidence": r.confidence},
                ))

        # Normalize and sort results across stores
        # TODO: Priority/ranking strategy across stores
```

**Aggregation patterns:**
- Each store returns different data types (vector similarity vs ES matches vs Zotero items)
- Create a common result format (MemorySearchResult) with optional fields
- Store-specific metadata in `metadata` dict (extensible)
- Document missing functionality (the TODO for ranking strategy)

### Step 6: Clean Module Exports

Provide a focused public API in `__init__.py`:

```python
# langchain_tools/__init__.py

from .base import StoreManager, get_store_manager
from .search_memory import (
    SearchMemoryTool,
    SearchMemoryInput,
    SearchMemoryOutput,
    MemorySearchResult,
)
from .expand_context import (
    ExpandContextTool,
    ExpandContextInput,
    ExpandedContext,
)

__all__ = [
    # Store management
    "StoreManager",
    "get_store_manager",
    # search_memory tool
    "SearchMemoryTool",
    "SearchMemoryInput",
    "SearchMemoryOutput",
    "MemorySearchResult",
    # expand_context tool
    "ExpandContextTool",
    "ExpandContextInput",
    "ExpandedContext",
]
```

**Benefits:**
- Internal structure hidden (can refactor without breaking imports)
- Consumers know exactly what's available
- Clear separation: management APIs vs tool classes vs schemas

## Complete Example

Full working integration in the Thala codebase:

```python
# Using search_memory tool in a LangChain agent
from langchain_tools import SearchMemoryTool, ExpandContextTool

# Create tools
search_tool = SearchMemoryTool()
expand_tool = ExpandContextTool()

# Use in agent
tools = [search_tool, expand_tool]

# When agent invokes search_memory:
# 1. Tool receives query: "What do I know about memory systems?"
# 2. StoreManager.embedding.embed(query) - lazy loads EmbeddingService
# 3. StoreManager.chroma.search() - lazy loads ChromaStore
# 4. Results aggregated and returned
# 5. Same manager instance reused for expand_context tool

# Tool automatically handles:
# - Environment configuration (THALA_ES_COHERENCE_HOST, etc.)
# - Lazy store initialization
# - Async execution
# - Cross-store result aggregation
# - Per-store error handling
```

## Consequences

### Benefits

- **Resource efficiency**: Stores only initialized when actually used
- **Clean testability**: Inject mock StoreManager in tests without complex mocking
- **Configuration simplicity**: All store hosts configured in one place via environment
- **Code reusability**: StoreManager can be shared across multiple tools
- **Graceful degradation**: One store failing doesn't break entire tool
- **Separation of concerns**: Base utilities separate from tool implementations
- **LLM-driven workflows**: Tools are tools, not APIs (LLM decides when to call)

### Trade-offs

- **Slightly more boilerplate**: Need StoreManager + BaseTool + async pattern
- **Async-only**: Synchronous code must wrap with `asyncio.run()`
- **Ordering dependencies**: Some stores depend on others (ChromaStore needs es_stores)
- **Global singleton**: Default manager instance could cause issues in multi-threaded context (mitigated by async-only design)

### Alternatives

1. **Direct dependency injection**: Pass all stores to tool __init__. Trade-off: more verbose, harder to test, initialization overhead.
2. **Tool factory**: Create tools with specific stores. Trade-off: can't share instances, more code.
3. **Sync wrapper pattern**: Make tools synchronous, wrap async stores. Trade-off: blocks on I/O, defeats LangChain's async benefits.
4. **Per-tool configuration**: Let each tool read its own env vars. Trade-off: configuration scattered, harder to override in tests.

## Related Patterns

- [Async-First Tool Design](./async-first-design.md) - Core async patterns this builds on
- [Environment-Driven Configuration](./environment-configuration.md) - How store configuration works

## Known Uses in Thala

- `/home/dave/thala/langchain_tools/search_memory.py`: Cross-store semantic search for LLM agents
- `/home/dave/thala/langchain_tools/expand_context.py`: Deep-dive retrieval with UUID/Zotero/fuzzy matching
- `/home/dave/thala/core/embedding.py`: Shared embedding service (moved from mcp_server)
- `/home/dave/thala/langchain_tools/base.py`: StoreManager implementation

## References

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
- [BaseTool Class](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.BaseTool.html)
- [Pydantic Field Documentation](https://docs.pydantic.dev/latest/concepts/fields/)
