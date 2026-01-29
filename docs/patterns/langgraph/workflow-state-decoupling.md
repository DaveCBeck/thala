---
name: workflow-state-decoupling
title: "Workflow State Decoupling via Direct Store Queries"
date: 2026-01-13
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/7fecce47dda5a1a046daac92d2ee223d
article_path: .context/libs/thala-dev/content/2026-01-13-workflow-state-decoupling-langgraph.md
applicability:
  - "Multi-phase LangGraph workflows where loops process stored data"
  - "Supervision systems with sequential editing loops"
  - "Workflows where paper/document corpus grows during execution"
  - "Systems needing standalone testable loop components"
components: [state_design, store_query_service, result_dataclass, orchestration]
complexity: moderate
verified_in_production: true
related_solutions:
  - quality-setting-propagation-max-papers
tags: [langgraph, state-management, decoupling, elasticsearch, store-queries, supervision-loops]
---

# Workflow State Decoupling via Direct Store Queries

## Intent

Simplify multi-phase workflow state by having loops query persistent stores directly for data, rather than passing large corpus dictionaries through state.

## Motivation

In multi-phase supervision workflows, earlier phases discover and store papers while later phases process them. The naive approach passes the full corpus through state:

```python
# ❌ BEFORE: State bloated with corpus data
class Loop5State(TypedDict):
    current_review: str
    paper_summaries: dict[str, Any]      # Full DOI -> summary mapping
    zotero_keys: dict[str, str]          # DOI -> key mapping
    # ... 10+ more fields

async def run_loop5_standalone(
    review: str,
    paper_summaries: dict,    # Passed from orchestration
    zotero_keys: dict,        # Full corpus mapping
    ...
) -> dict:
    # Loop filters and re-processes corpus data
```

This creates problems:
- **Tight coupling**: Loops depend on orchestration's data structures
- **State bloat**: Large dicts serialized through every node transition
- **Stale data**: Corpus reflects discovery-time state, not current store contents
- **Testing difficulty**: Loops require realistic corpus fixtures

## Applicability

Use this pattern when:
- Workflows have multiple sequential phases processing stored documents
- Data is persisted to external stores (Elasticsearch, Zotero, ChromaDB)
- Later phases only need to query subsets of stored data
- Loops should be independently testable without full pipeline setup

Do NOT use this pattern when:
- All data exists only in workflow state (no persistent stores)
- Phases genuinely need the full corpus in memory for cross-referencing
- Query latency would significantly impact hot loops
- State data requires complex joins across multiple sources

## Structure

```
BEFORE: Corpus Data Flows Through State
┌─────────────────────────────────────────────────────────┐
│  Orchestrator                                           │
│  ┌─────────┐   corpus   ┌─────────┐   corpus   ┌─────┐ │
│  │ Loop 1  │ ──────────▶│ Loop 2  │ ──────────▶│ ... │ │
│  └─────────┘  summaries └─────────┘  summaries └─────┘ │
│               zotero_keys           zotero_keys         │
└─────────────────────────────────────────────────────────┘

AFTER: Loops Query Stores Directly
┌─────────────────────────────────────────────────────────┐
│  Orchestrator (passes minimal params)                   │
│  ┌─────────┐  review   ┌─────────┐  review   ┌───────┐ │
│  │ Loop 1  │ ─────────▶│ Loop 2  │ ─────────▶│  ...  │ │
│  └────┬────┘  topic    └────┬────┘  topic    └───┬───┘ │
│       │       quality       │       quality      │     │
│       ▼                     ▼                    ▼     │
│  ┌─────────────────────────────────────────────────┐   │
│  │            StoreQueryService                    │   │
│  │  ES (papers)  │  Zotero (metadata)  │  Chroma   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Create Store Query Service

Centralize store access with lazy initialization and fallback strategies:

```python
"""Store query service for supervision loops."""

from typing import Any, Optional
from langchain_tools.base import get_store_manager, StoreManager
from core.stores import ZoteroStore


class SupervisionStoreQuery:
    """Query service for loops to access paper content directly.

    Enables loops to fetch paper content at different compression levels:
    - L0: Full original document
    - L1: Short summary (~100 words)
    - L2: 10:1 compressed summary

    Queries Elasticsearch directly by zotero_key.
    """

    def __init__(self):
        self._store_manager: Optional[StoreManager] = None
        self._zotero_client: Optional[ZoteroStore] = None

    @property
    def store_manager(self) -> StoreManager:
        """Lazy initialization of store manager."""
        if self._store_manager is None:
            self._store_manager = get_store_manager()
        return self._store_manager

    async def get_paper_content(
        self,
        zotero_key: str,
        compression_level: int = 2,
    ) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Fetch paper content by zotero_key with compression fallback."""
        store = self.store_manager.es_stores.store

        query = {"term": {"zotero_key": zotero_key}}
        records = await store.search(
            query=query,
            size=1,
            compression_level=compression_level,
        )

        if records:
            return records[0].content, records[0].metadata

        # Fallback to other compression levels
        for alt_level in [1, 2, 0]:
            if alt_level == compression_level:
                continue
            records = await store.search(
                query=query, size=1, compression_level=alt_level
            )
            if records:
                return records[0].content, records[0].metadata

        return None, None

    async def close(self):
        """Close connections."""
        if self._zotero_client is not None:
            await self._zotero_client.close()
```

### Step 2: Define Minimal Loop State

Remove corpus data from state, keep only iteration control:

```python
from dataclasses import dataclass
from typing_extensions import TypedDict


class Loop2State(TypedDict, total=False):
    """State for Loop 2 - NO corpus data passed through.

    Papers flow directly to Elasticsearch via nested workflows.
    """

    # Core inputs (minimal)
    current_review: str
    topic: str
    research_questions: list[str]
    quality_settings: dict

    # Iteration tracking
    iteration: int
    max_iterations: int
    explored_bases: list[str]
    is_complete: bool

    # Node outputs
    decision: dict | None

    # Error tracking
    errors: list[dict]


@dataclass
class Loop2Result:
    """Typed result from Loop 2."""

    current_review: str
    changes_summary: str
    explored_bases: list[str]
```

### Step 3: Simplify Standalone Loop Signatures

Loops accept only essential parameters:

```python
from langsmith import traceable


@traceable(run_type="chain", name="Loop2_LiteratureExpansion")
async def run_loop2_standalone(
    review: str,
    topic: str,
    research_questions: list[str],
    quality_settings: dict,
    config: dict | None = None,
) -> Loop2Result:
    """Run Loop 2 literature base expansion.

    Papers discovered are sent to Elasticsearch/Zotero via nested workflow.
    No corpus data passed back to orchestration.
    """
    max_iterations = quality_settings.get("max_stages", 1)

    initial_state: Loop2State = {
        "current_review": review,
        "topic": topic,
        "research_questions": research_questions,
        "quality_settings": quality_settings,
        "iteration": 0,
        "max_iterations": max_iterations,
        "explored_bases": [],
        "is_complete": False,
    }

    graph = create_loop2_graph()
    result = await graph.ainvoke(initial_state, config=config)

    return Loop2Result(
        current_review=result["current_review"],
        changes_summary=f"Expanded {len(result['explored_bases'])} literature bases",
        explored_bases=result.get("explored_bases", []),
    )
```

### Step 4: Update Orchestration Nodes

Orchestration passes minimal params and extracts typed results:

```python
async def run_loop2_node(state: OrchestrationState) -> dict[str, Any]:
    """Run Loop 2 - pass minimal params, extract typed result."""
    input_data = state["input"]

    result = await run_loop2_standalone(
        review=state["current_review"],
        topic=input_data.get("topic", ""),
        research_questions=input_data.get("research_questions", []),
        quality_settings=state["quality_settings"],
        config={"run_name": f"loop2:{input_data.get('topic', '')[:20]}"},
    )

    # No corpus merging - papers already in stores
    return {
        "current_review": result.current_review,
        "review_loop2": result.current_review,
        "loop2_result": {
            "explored_bases": result.explored_bases,
            "changes_summary": result.changes_summary,
        },
    }
```

## Complete Example

```python
"""Citation verification using store queries instead of corpus state."""

from workflows.wrappers.supervised_lit_review.supervision.store_query import (
    SupervisionStoreQuery,
)


async def resolve_invalid_citations(
    document: str,
    invalid_keys: set[str],
    topic: str,
) -> str:
    """Resolve invalid citations by querying stores directly.

    Instead of checking against a passed-in corpus dict, queries
    Elasticsearch and Zotero directly for verification and resolution.
    """
    store_query = SupervisionStoreQuery()

    try:
        for key in invalid_keys:
            # Query store directly instead of filtering passed-in dict
            content, metadata = await store_query.get_paper_content(key)

            if content and metadata:
                # Key exists in store - citation is valid
                continue

            # Key not in store - use LLM to resolve
            document = await _llm_resolve_citation(
                document=document,
                invalid_key=key,
                topic=topic,
                store_query=store_query,
            )

        return document

    finally:
        await store_query.close()
```

## Consequences

### Benefits

- **Decoupling**: Loops don't depend on orchestration's data structures
- **Scalability**: State size independent of corpus size
- **Fresh data**: Queries return current store contents, not stale snapshots
- **Testability**: Loops runnable standalone with mock store queries
- **Type safety**: Typed `Result` dataclasses instead of untyped dicts
- **Independent evolution**: Loops can change without orchestration updates

### Trade-offs

- **Query latency**: Each store query adds network overhead
- **Connection management**: Must properly close store clients
- **Cache coordination**: No automatic deduplication of repeated queries
- **Debugging complexity**: Data flow less visible in state snapshots

### Alternatives

- **State checkpointing**: Snapshot full state at phase boundaries for recovery
- **Hybrid approach**: Pass small lookup dicts (key→id), query content on demand
- **Materialized views**: Precompute filtered corpus views before loop execution

## Related Patterns

- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - State flow through sequential loops
- [Workflow Modularization Pattern](./workflow-modularization-pattern.md) - Core/wrapper composition
- [Deep Research Workflow Architecture](./deep-research-workflow-architecture.md) - Reducers for parallel writes
- [Compression-Level Based Index Routing](../stores/compression-level-index-routing.md) - Multi-level store queries

## Related Solutions

- [Quality Setting Propagation: max_papers](../../solutions/workflow-issues/quality-setting-propagation-max-papers.md) - Shows problem this pattern solves

## Known Uses in Thala

- `workflows/wrappers/supervised_lit_review/supervision/store_query.py`: SupervisionStoreQuery service
- `workflows/wrappers/supervised_lit_review/supervision/loops/loop2/graph.py`: Loop2State, Loop2Result
- `workflows/wrappers/supervised_lit_review/supervision/loops/loop5/graph.py`: Loop5State, Loop5Result
- `workflows/wrappers/supervised_lit_review/supervision/orchestration/nodes.py`: Simplified orchestration nodes
- `core/stores/utils.py`: verify_zotero_keys utility

## References

- [LangGraph State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [TypedDict for State Typing](https://docs.python.org/3/library/typing.html#typing.TypedDict)
