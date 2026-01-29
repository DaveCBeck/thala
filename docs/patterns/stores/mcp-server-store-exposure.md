---
name: mcp-server-store-exposure
title: "MCP Server Pattern for Store API Exposure"
date: 2025-12-17
category: stores
applicability:
  - "When exposing multiple data stores to LLM tool use"
  - "When different stores need different access levels (read-only vs CRUD)"
  - "When embedding generation must be coordinated with store operations"
components: [mcp_tool, elasticsearch, chroma, zotero, embedding]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [mcp, store, api, tool-use, embedding, crud]
---

# MCP Server Pattern for Store API Exposure

## Intent

Expose multiple heterogeneous data stores through a unified MCP server with consistent tool naming, access control, and coordinated embedding generation.

## Motivation

When building an LLM-accessible knowledge system with multiple backends (Elasticsearch, ChromaDB, Zotero), you need:
- Consistent tool naming conventions across stores
- Different access levels (read-only archives vs. full CRUD)
- Coordinated embedding generation for stores that need it
- Actionable error messages that help the LLM self-correct

## Applicability

Use this pattern when:
- You have 3+ data stores with different backends
- Some stores need full CRUD while others are read-only
- Embedding generation must happen on write operations
- Tools will be used by LLMs that need clear error guidance

Do NOT use this pattern when:
- You have a single store (use direct API instead)
- All stores have identical access patterns
- Embedding is handled externally

## Structure

```
mcp_server/
├── __init__.py          # Entry point
├── server.py            # MCP server setup, routing, store init
├── embedding.py         # Configurable embedding service
├── errors.py            # Typed errors with actionable messages
└── tools/
    ├── __init__.py
    ├── health.py        # Connectivity checks
    ├── coherence.py     # CRUD + embedding
    ├── store.py         # CRUD + embedding
    ├── top_of_mind.py   # CRUD + embedding + semantic search
    ├── zotero.py        # CRUD (no embedding)
    ├── who_i_was.py     # Read-only (history)
    └── forgotten.py     # Read-only (archives)
```

## Implementation

### Step 1: Define Error Types with Actionable Messages

Errors should tell the LLM how to recover:

```python
class NotFoundError(ToolError):
    """Record not found in store."""

    def __init__(self, store: str, record_id: str):
        super().__init__(
            f"Record not found. The UUID '{record_id}' does not exist in {store} store. "
            f"Use {store}.search to find records, or verify the UUID is correct.",
            {"store": store, "record_id": record_id},
        )
```

### Step 2: Consistent Tool Naming Convention

All tools follow `{store_name}.{operation}` pattern:

```python
def get_tools() -> list[Tool]:
    return [
        Tool(
            name="coherence.add",
            description="Add a new coherence record...",
            inputSchema={...},
        ),
        Tool(
            name="coherence.get",
            description="Retrieve a coherence record by UUID.",
            inputSchema={...},
        ),
        Tool(
            name="coherence.update",
            description="Update a coherence record...",
            inputSchema={...},
        ),
        Tool(
            name="coherence.delete",
            description="Delete a coherence record...",
            inputSchema={...},
        ),
        Tool(
            name="coherence.search",
            description="Search coherence records...",
            inputSchema={...},
        ),
    ]
```

### Step 3: Centralized Tool Routing

Route calls by prefix to maintain separation:

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name.startswith("health."):
            result = await health.handle(name, arguments, _stores, _embedding_service)
        elif name.startswith("who_i_was."):
            result = await who_i_was.handle(name, arguments, _stores)
        elif name.startswith("coherence."):
            result = await coherence.handle(name, arguments, _stores, _embedding_service)
        # ... other stores
        else:
            raise ToolError(f"Unknown tool: {name}. Use health.check to see available tools.")

        return [TextContent(type="text", text=json.dumps(result, default=str))]

    except ToolError as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": e.message, "details": e.details}),
        )]
```

### Step 4: Configurable Embedding Service

Support multiple providers with consistent interface:

```python
class EmbeddingService:
    """
    Configuration via environment variables:
    - THALA_EMBEDDING_PROVIDER: 'openai' or 'ollama'
    - THALA_EMBEDDING_MODEL: model name
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        provider = provider or os.environ.get("THALA_EMBEDDING_PROVIDER", "openai")

        if provider == "openai":
            model = model or os.environ.get("THALA_EMBEDDING_MODEL", "text-embedding-3-small")
            self._provider = OpenAIEmbeddings(model=model)
        elif provider == "ollama":
            model = model or os.environ.get("THALA_EMBEDDING_MODEL", "nomic-embed-text")
            self._provider = OllamaEmbeddings(model=model)

    async def embed(self, text: str) -> list[float]:
        return await self._provider.embed(text)
```

### Step 5: Access Level Enforcement

Read-only tools only get `get` and `search`:

```python
# who_i_was.py - Read-only
def get_tools() -> list[Tool]:
    return [
        Tool(name="who_i_was.get", ...),
        Tool(name="who_i_was.get_history", ...),  # Extra: history traversal
        Tool(name="who_i_was.search", ...),
    ]

# coherence.py - Full CRUD
def get_tools() -> list[Tool]:
    return [
        Tool(name="coherence.add", ...),
        Tool(name="coherence.get", ...),
        Tool(name="coherence.update", ...),
        Tool(name="coherence.delete", ...),
        Tool(name="coherence.search", ...),
    ]
```

## Complete Example

See `mcp_server/` in the Thala codebase for the full 27-tool implementation.

## Consequences

### Benefits

- **Discoverability**: LLMs can list all tools and understand the naming convention
- **Consistency**: Same patterns across all stores reduce cognitive load
- **Graceful degradation**: Stores can fail independently without crashing the server
- **Actionable errors**: LLMs can self-correct based on error messages

### Trade-offs

- **Boilerplate**: Each store needs similar get/search/add/update/delete handlers
- **Coupling**: Embedding service is tightly coupled to write operations
- **Memory**: All stores initialized at startup even if unused

### Alternatives

- **Single unified store**: If all data fits one schema, skip the multi-store pattern
- **REST API**: If tools won't be used by LLMs, standard REST may be simpler
- **Direct client access**: If only one consumer, skip the MCP abstraction

## Related Patterns

- [Mandatory Archive Before Delete](./mandatory-archive-before-delete.md)
- [Compression-Level Index Routing](./compression-level-index-routing.md) - Tiered storage for document originals and summaries

## Known Uses in Thala

- `mcp_server/server.py`: Server initialization and tool routing
- `mcp_server/tools/coherence.py`: Full CRUD with embedding for beliefs/preferences
- `mcp_server/tools/top_of_mind.py`: Vector store with semantic search
- `mcp_server/tools/who_i_was.py`: Read-only history access

## References

- [Model Context Protocol specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
