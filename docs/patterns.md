# Patterns

Common patterns used across the Thala codebase.

## State Management

TypedDict-based state with reducer functions for merging updates:

```python
from typing import TypedDict, Annotated
from operator import add

def merge_paper_summaries(existing: dict, new: dict) -> dict:
    """Custom reducer for paper dictionaries."""
    return {**existing, **new}

class WorkflowState(TypedDict):
    papers: Annotated[dict, merge_paper_summaries]  # Merges new papers
    findings: Annotated[list, add]                  # Appends to list
    iteration: int                                  # Overwrites
```

## Quality Tiers

Consistent quality presets across all workflows:

```python
QUALITY_PRESETS: dict[str, QualitySettings] = {
    "test": QualitySettings(max_stages=1, max_papers=10),
    "quick": QualitySettings(max_stages=2, max_papers=50),
    "standard": QualitySettings(max_stages=3, max_papers=100),
    "comprehensive": QualitySettings(max_stages=5, max_papers=200),
}

# Usage
settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
```

## Lazy-Initialized Singletons

Stores and managers initialized on first access:

```python
class StoreManager:
    _instance = None

    def __init__(self):
        self._es_stores = None
        self._chroma = None

    @property
    def elasticsearch(self) -> ElasticsearchStores:
        if self._es_stores is None:
            self._es_stores = ElasticsearchStores()
        return self._es_stores

_default_manager = None

def get_store_manager() -> StoreManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = StoreManager()
    return _default_manager
```

## Structured Output

Auto-selecting strategy for LLM responses:

```python
from workflows.shared.llm_utils import get_structured_output, ModelTier

result = await get_structured_output(
    output_schema=PaperAnalysis,
    user_prompt="Analyze this paper",
    tier=ModelTier.SONNET,
    # Auto-selects: LANGCHAIN → BATCH → AGENT → JSON
)
```

Strategies tried in order:
1. **LangChain** - Native structured output
2. **Batch API** - Cost-efficient bulk processing
3. **Agent** - Tool-based extraction
4. **JSON** - Parse from response text

## Caching

### Persistent Cache (File-based)

```python
from workflows.shared.persistent_cache import PersistentCache

cache = PersistentCache(namespace="embeddings", ttl_days=90)
result = await cache.get_or_compute(key, expensive_operation)
```

### TTL Cache (In-memory)

```python
from workflows.shared.ttl_cache import TTLCache

cache = TTLCache(ttl_seconds=300)
cache.set("key", value)
value = cache.get("key")  # None if expired
```

## Error Handling

### Transient vs Permanent Errors

```python
def _is_transient_error(error: Exception) -> bool:
    error_str = str(error).lower()
    transient_patterns = ["timeout", "rate limit", "connection", "503"]
    return any(p in error_str for p in transient_patterns)
```

### Retry with Backoff

```python
from workflows.shared.retry_utils import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=1.0)
async def fetch_data():
    return await client.get(url)
```

## Multi-Language Support

ISO 639-1 codes throughout with translation layer:

```python
from workflows.shared.language import translate_query, SUPPORTED_LANGUAGES

# 30+ supported languages
if language_code in SUPPORTED_LANGUAGES:
    translated = await translate_query(query, target_lang=language_code)
```

## Workflow Registry

Dynamic workflow registration and discovery:

```python
from workflows.shared.wrappers import register_workflow, get_workflow

register_workflow(
    name="my_workflow",
    callable=my_workflow_function,
    quality_tiers=["standard", "comprehensive"],
    description="Does something useful",
)

# Later
workflow = get_workflow("my_workflow")
result = await workflow(input_data)
```

## Subgraph Composition

LangGraph workflows composed from reusable subgraphs:

```python
from langgraph.graph import StateGraph

graph = StateGraph(MyState)
graph.add_node("step1", subgraph_a)
graph.add_node("step2", subgraph_b)
graph.add_conditional_edges(
    "step1",
    lambda s: "continue" if s["should_continue"] else "end",
    {"continue": "step2", "end": END}
)
```

## Service Configuration

Environment-driven with defaults:

```python
import os

host = os.environ.get("THALA_ES_COHERENCE_HOST", "http://localhost:9201")
port = int(os.environ.get("THALA_CHROMA_PORT", "8000"))

# All services support:
# - Host/port override
# - Health checks
# - Async cleanup
```

## Async Context Managers

Proper resource cleanup:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_client():
    client = await create_client()
    try:
        yield client
    finally:
        await client.close()

async with managed_client() as client:
    result = await client.fetch()
```

## Safe Node Execution

Workflow nodes with error handling:

```python
from workflows.shared.node_utils import safe_node

@safe_node
async def process_step(state: MyState) -> dict:
    # If exception raised, returns partial state update
    # with error information
    result = await expensive_operation(state["input"])
    return {"output": result}
```
