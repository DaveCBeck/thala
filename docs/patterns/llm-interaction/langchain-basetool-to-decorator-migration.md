---
name: langchain-basetool-to-decorator-migration
title: "Migrating LangChain Tools from BaseTool to @tool Decorator"
date: 2025-12-17
category: llm-interaction
applicability:
  - "When upgrading LangChain tools to v1.x patterns"
  - "When refactoring class-based tools to function-based tools"
  - "When simplifying tool implementations that don't need class state"
components: [llm_call, async_task]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [langchain, tools, migration, refactoring, decorator, basetool]
---

# Migrating LangChain Tools from BaseTool to @tool Decorator

## Intent

Migrate LangChain tools from the legacy `BaseTool` class inheritance pattern to the modern `@tool` decorator pattern recommended for LangChain 1.x.

## Motivation

LangChain 1.x recommends using the `@tool` decorator over `BaseTool` class inheritance because:

1. **Less boilerplate**: No need for `args_schema`, `Config`, `_run`/`_arun` method pairs
2. **Cleaner async**: Just `async def` - no sync wrapper that raises NotImplementedError
3. **Better type inference**: Function parameters become the schema automatically
4. **Simpler imports**: `from langchain.tools import tool` vs multiple base class imports
5. **Docstring as description**: The function docstring becomes the tool description

This pattern documents how to perform this migration systematically.

## Applicability

Use this pattern when:
- Migrating existing BaseTool classes to LangChain 1.x
- Tools don't need instance state between calls
- You want simpler, more maintainable tool code
- Converting async-only tools (no sync implementation needed)

Do NOT use this pattern when:
- Tools genuinely need class state between calls
- You need custom Tool configuration beyond what @tool provides
- Tools have complex initialization that benefits from __init__

## Structure

### Before: BaseTool Class Pattern

```
┌─────────────────────────────────────────────┐
│  MyTool(BaseTool)                           │
├─────────────────────────────────────────────┤
│  name: str = "my_tool"                      │
│  description: str = "..."                   │
│  args_schema: type[BaseModel] = MyInput     │
│  store_manager: StoreManager = Field(...)   │
├─────────────────────────────────────────────┤
│  class Config:                              │
│      arbitrary_types_allowed = True         │
├─────────────────────────────────────────────┤
│  def _run(...) -> ...:                      │
│      raise NotImplementedError(...)         │
│                                             │
│  async def _arun(...) -> ...:               │
│      # actual implementation                │
└─────────────────────────────────────────────┘
```

### After: @tool Decorator Pattern

```
┌─────────────────────────────────────────────┐
│  @tool                                      │
│  async def my_tool(                         │
│      param1: str,                           │
│      param2: int = 10,                      │
│  ) -> dict:                                 │
│      """Description from docstring."""      │
│      store_manager = get_store_manager()    │
│      # implementation                       │
│      return result                          │
└─────────────────────────────────────────────┘
```

## Implementation

### Step 1: Extract Input Parameters from args_schema

**Before**: Separate Pydantic model for input schema

```python
class SearchMemoryInput(BaseModel):
    """Input schema for search_memory tool."""

    query: str = Field(description="What to search for in memory")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max results per store (default 10)",
    )
    stores: Optional[list[Literal["top_of_mind", "coherence", "store"]]] = Field(
        default=None,
        description="Specific stores to search. Defaults to all.",
    )
```

**After**: Parameters become function arguments with defaults

```python
@tool
async def search_memory(
    query: str,
    limit: int = 10,
    stores: Optional[list[Literal["top_of_mind", "coherence", "store"]]] = None,
) -> dict:
```

**Migration notes**:
- Field descriptions move to the docstring Args section
- Field constraints (ge, le) are lost unless you add runtime validation
- Consider keeping the Input class if you need complex validation

### Step 2: Convert Class to Function with @tool Decorator

**Before**: BaseTool class with multiple methods

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SearchMemoryTool(BaseTool):
    """Search across the memory system for relevant information."""

    name: str = "search_memory"
    description: str = """Search across the memory system...

Use this when you need to:
- Recall something you might have learned before
- Find beliefs, preferences, or identity information"""

    args_schema: type[BaseModel] = SearchMemoryInput

    store_manager: StoreManager = Field(default_factory=get_store_manager)

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        query: str,
        limit: int = 10,
        stores: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Sync wrapper - raises error, use async version."""
        raise NotImplementedError("SearchMemoryTool only supports async.")

    async def _arun(
        self,
        query: str,
        limit: int = 10,
        stores: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Async implementation."""
        # ... implementation using self.store_manager
```

**After**: Simple async function with @tool decorator

```python
from langchain.tools import tool

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

    Args:
        query: What to search for in memory
        limit: Max results per store (default 10, max 50)
        stores: Specific stores to search. Defaults to all.
    """
    store_manager = get_store_manager()
    # ... implementation using store_manager
```

### Step 3: Move Instance Dependencies to Function Body

**Before**: Dependencies injected via Pydantic Field

```python
class ExpandContextTool(BaseTool):
    store_manager: StoreManager = Field(default_factory=get_store_manager)

    async def _arun(self, reference: str, ...) -> dict:
        record = await self.store_manager.es_stores.store.get(record_id)
```

**After**: Dependencies accessed via factory function

```python
@tool
async def expand_context(reference: str, ...) -> dict:
    store_manager = get_store_manager()
    record = await store_manager.es_stores.store.get(record_id)
```

**Why this works**:
- `get_store_manager()` returns a singleton - same instance every call
- No performance penalty (lazy init already done)
- Simpler code, easier to test by mocking the factory

### Step 4: Extract Helper Methods to Module-Level Functions

**Before**: Private methods on the class

```python
class ExpandContextTool(BaseTool):
    def _detect_reference_type(self, reference: str, hint: str) -> str:
        """Auto-detect reference type based on format."""
        if hint != "auto":
            return hint
        if UUID_PATTERN.match(reference):
            return "uuid"
        # ...

    async def _expand_by_uuid(self, reference: str, result: ExpandedContext, ...) -> None:
        """Look up record by UUID across all stores."""
        record = await self.store_manager.es_stores.store.get(record_id)
        # ...
```

**After**: Module-level private functions

```python
def _detect_reference_type(reference: str, hint: str) -> str:
    """Auto-detect reference type based on format."""
    if hint != "auto":
        return hint
    if UUID_PATTERN.match(reference):
        return "uuid"
    # ...


async def _expand_by_uuid(
    reference: str,
    result: ExpandedContext,
    include_zotero: bool,
    include_history: bool,
) -> None:
    """Look up record by UUID across all stores."""
    store_manager = get_store_manager()
    record = await store_manager.es_stores.store.get(record_id)
    # ...


@tool
async def expand_context(reference: str, ...) -> dict:
    detected_type = _detect_reference_type(reference, reference_type)
    if detected_type == "uuid":
        await _expand_by_uuid(reference, result, include_zotero, include_history)
```

### Step 5: Update Exports

**Before**: Export class names

```python
# langchain_tools/__init__.py
from .search_memory import (
    SearchMemoryTool,
    SearchMemoryInput,
    SearchMemoryOutput,
)

__all__ = [
    "SearchMemoryTool",
    "SearchMemoryInput",
    "SearchMemoryOutput",
]
```

**After**: Export function names (no Input class needed)

```python
# langchain_tools/__init__.py
from .search_memory import (
    search_memory,
    MemorySearchResult,
    SearchMemoryOutput,
)

__all__ = [
    "search_memory",
    "MemorySearchResult",
    "SearchMemoryOutput",
]
```

### Step 6: Update Usage Sites

**Before**: Instantiate tool classes

```python
from langchain_tools import SearchMemoryTool, ExpandContextTool

tools = [SearchMemoryTool(), ExpandContextTool()]
```

**After**: Use functions directly

```python
from langchain_tools import search_memory, expand_context

tools = [search_memory, expand_context]
```

**Also update imports** (dependency change):

```python
# requirements.txt
# Before:
langchain-core>=0.3.0

# After:
langchain>=1.2.0
```

## Complete Example

### Before (BaseTool Pattern)

```python
"""langchain_tools/search_memory.py - BEFORE"""

from typing import Any, Literal, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from .base import StoreManager, get_store_manager


class SearchMemoryInput(BaseModel):
    query: str = Field(description="What to search for")
    limit: int = Field(default=10, ge=1, le=50)


class SearchMemoryTool(BaseTool):
    name: str = "search_memory"
    description: str = "Search across the memory system..."
    args_schema: type[BaseModel] = SearchMemoryInput
    store_manager: StoreManager = Field(default_factory=get_store_manager)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, limit: int = 10) -> dict[str, Any]:
        raise NotImplementedError("Use ainvoke()")

    async def _arun(self, query: str, limit: int = 10) -> dict[str, Any]:
        results = []
        query_embedding = await self.store_manager.embedding.embed(query)
        chroma_results = await self.store_manager.chroma.search(
            query_embedding=query_embedding,
            n_results=limit,
        )
        # ... process results
        return {"query": query, "results": results}
```

### After (@tool Decorator Pattern)

```python
"""langchain_tools/search_memory.py - AFTER"""

from typing import Literal, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
from .base import get_store_manager


class MemorySearchResult(BaseModel):
    id: str
    source_store: str
    content: str
    score: Optional[float] = None


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

    Args:
        query: What to search for in memory
        limit: Max results per store (default 10, max 50)
        stores: Specific stores to search. Defaults to all.
    """
    store_manager = get_store_manager()
    results = []

    limit = min(max(1, limit), 50)  # Enforce bounds here instead of Field

    query_embedding = await store_manager.embedding.embed(query)
    chroma_results = await store_manager.chroma.search(
        query_embedding=query_embedding,
        n_results=limit,
    )
    # ... process results

    return {"query": query, "total_results": len(results), "results": results}
```

## Consequences

### Benefits

- **~50% less code**: No Input class, no Config, no _run stub, no Field declarations
- **Clearer structure**: Single function with docstring instead of class with multiple pieces
- **Modern pattern**: Aligns with LangChain 1.x recommendations
- **Better IDE support**: Function signatures work better with autocomplete
- **Simpler testing**: Test a function, not a class with configuration

### Trade-offs

- **Lost Field constraints**: `ge=1, le=50` must be enforced in function body
- **No instance state**: Can't store computed values between calls (usually not needed)
- **Different import**: `from langchain.tools import tool` vs `from langchain_core.tools import BaseTool`

### Alternatives

- **Keep BaseTool**: If you genuinely need class state or complex initialization
- **StructuredTool.from_function**: Middle ground with more configuration options
- **Custom Tool class**: For tools that need both sync and async implementations

## Related Patterns

- [LangChain Tools Integration](./langchain-tools-integration.md): Original BaseTool pattern (legacy)
- [LangChain Tools Store Integration](./langchain-tools-store-integration.md): @tool pattern with store access

## Known Uses in Thala

- `langchain_tools/search_memory.py`: Migrated from SearchMemoryTool class
- `langchain_tools/expand_context.py`: Migrated from ExpandContextTool class

## References

- [LangChain 1.x Migration Guide](https://python.langchain.com/docs/versions/migrating_chains/)
- [LangChain @tool Decorator](https://python.langchain.com/docs/how_to/custom_tools/)
- Commit `bcf17fa`: Original migration implementing this pattern
