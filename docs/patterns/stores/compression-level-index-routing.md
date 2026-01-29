---
name: compression-level-index-routing
title: "Compression-Level Based Index Routing for Tiered Document Storage"
date: 2025-12-18
category: stores
applicability:
  - "Multi-resolution document storage with originals and summaries at different compression ratios"
  - "Hybrid search systems requiring both full-text and semantic vector search"
  - "Storage optimization where only derived/summarized content needs embeddings"
components: [elasticsearch, embedding, workflow_graph]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [elasticsearch, vector-search, data-tiering, index-routing, embeddings, knn, document-processing]
---

# Compression-Level Based Index Routing for Tiered Document Storage

## Intent

Automatically route document records to specialized Elasticsearch indices based on their compression level, enabling optimized storage and search strategies for different content tiers (originals vs. summaries).

## Motivation

Document processing pipelines often create multiple representations of the same content:
- **Original documents (L0)**: Full text, large, primarily for full-text search
- **Short summaries (L1)**: ~500 words, optimized for quick retrieval
- **Compressed summaries (L2)**: 10:1 ratio, highly semantic-dense

Storing all these in a single index creates problems:
1. **Wasted storage**: Embeddings for full documents are expensive and less useful than summary embeddings
2. **Schema conflicts**: Not all levels need the same fields (e.g., L0 doesn't need vector fields)
3. **Query inefficiency**: Vector search on L0 is pointless since it lacks embeddings
4. **Performance issues**: Large text documents bloat vector indices

This pattern solves these issues by routing records to level-specific indices with optimized schemas.

## Applicability

Use this pattern when:
- You have multi-resolution document representations (originals + summaries)
- Different content tiers have different search requirements (text vs. semantic)
- You want to optimize storage by only embedding summarized content
- You need hierarchical document retrieval at different fidelity levels

Do NOT use this pattern when:
- All documents have identical structure and search requirements
- You only store one representation per document
- The overhead of multiple indices outweighs the benefits (small datasets)

## Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                        MainStore API                            │
│  store.add(record)  →  Routes by record.compression_level       │
│  store.search(query, compression_level=None)  →  store_l*       │
│  store.knn_search(embedding, compression_level=1)  →  store_l1  │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ store_l0  │   │ store_l1  │   │ store_l2  │
        │───────────│   │───────────│   │───────────│
        │ content   │   │ content   │   │ content   │
        │ metadata  │   │ metadata  │   │ metadata  │
        │ (no vec)  │   │ embedding │   │ embedding │
        │───────────│   │ 1536-dim  │   │ 1536-dim  │
        │ Full-text │   │ cosine    │   │ cosine    │
        │ search    │   │───────────│   │───────────│
        └───────────┘   │ Text+KNN  │   │ Text+KNN  │
                        └───────────┘   └───────────┘
```

## Implementation

### Step 1: Define Index Templates

Create specialized Elasticsearch index templates for each compression level.

**store_l0.json (no embeddings):**
```json
{
  "index_patterns": ["store_l0"],
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 0
    },
    "mappings": {
      "properties": {
        "id": { "type": "keyword" },
        "content": { "type": "text" },
        "compression_level": { "type": "integer" },
        "source_ids": { "type": "keyword" },
        "created_at": { "type": "date" },
        "updated_at": { "type": "date" }
      }
    }
  }
}
```

**store_l1.json / store_l2.json (with embeddings):**
```json
{
  "index_patterns": ["store_l1"],
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 0
    },
    "mappings": {
      "properties": {
        "id": { "type": "keyword" },
        "content": { "type": "text" },
        "compression_level": { "type": "integer" },
        "source_ids": { "type": "keyword" },
        "embedding": {
          "type": "dense_vector",
          "dims": 1536,
          "index": true,
          "similarity": "cosine"
        },
        "embedding_model": { "type": "keyword" },
        "created_at": { "type": "date" },
        "updated_at": { "type": "date" }
      }
    }
  }
}
```

### Step 2: Implement Index Routing in Store Class

Override the base store's index selection to route based on record properties.

```python
class MainStore(BaseElasticsearchStore):
    """
    Store for document content with compression-level based routing.

    Routes to different indices based on compression_level:
    - store_l0: Original documents (compression_level=0)
    - store_l1: Short summaries (compression_level=1)
    - store_l2: 10:1 summaries (compression_level=2)
    """

    index_name = "store_l0"  # Default for backwards compatibility
    record_class = StoreRecord

    # Index mapping by compression level
    COMPRESSION_INDICES = {
        0: "store_l0",
        1: "store_l1",
        2: "store_l2",
    }

    def _get_index_name(self, record: Optional[BaseRecord] = None) -> str:
        """Route to correct index based on compression_level."""
        if record is None:
            return self.index_name
        if isinstance(record, StoreRecord):
            return self.COMPRESSION_INDICES.get(record.compression_level, "store_l0")
        return self.index_name

    def _index_for_level(self, compression_level: int) -> str:
        """Get index name for a compression level."""
        return self.COMPRESSION_INDICES.get(compression_level, "store_l0")
```

### Step 3: Implement Level-Aware CRUD Operations

Override methods to accept optional `compression_level` for optimized lookups.

```python
async def get(self, record_id: UUID, compression_level: Optional[int] = None) -> Optional[StoreRecord]:
    """
    Get a record by UUID.

    Args:
        record_id: UUID of the record
        compression_level: If known, speeds up lookup. Otherwise searches all indices.
    """
    if compression_level is not None:
        index = self._index_for_level(compression_level)
        return await super().get(record_id, index=index)

    # Search across all store indices
    for level in self.COMPRESSION_INDICES.values():
        result = await super().get(record_id, index=level)
        if result:
            return result
    return None

async def search(
    self,
    query: dict[str, Any],
    size: int = 10,
    compression_level: Optional[int] = None,
) -> list[StoreRecord]:
    """
    Search for records.

    Args:
        query: Elasticsearch query DSL
        size: Max results to return
        compression_level: If specified, search only that level. Otherwise search all.
    """
    if compression_level is not None:
        index = self._index_for_level(compression_level)
    else:
        # Search across all store indices using wildcard
        index = "store_l*"

    return await super().search(query, size, index=index)
```

### Step 4: Implement Vector Search with Safety Guards

Add KNN search that only operates on embedding-enabled indices.

```python
async def knn_search(
    self,
    embedding: list[float],
    k: int = 10,
    compression_level: Optional[int] = None,
    num_candidates: int = 100,
) -> list[tuple[StoreRecord, float]]:
    """
    Perform KNN vector search on summaries.

    Args:
        embedding: Query embedding vector
        k: Number of results to return
        compression_level: 1 or 2 (l0 has no embeddings). None searches both.
        num_candidates: Number of candidates to consider

    Returns:
        List of (record, score) tuples sorted by similarity

    Raises:
        ValueError: If compression_level=0 (L0 has no embeddings)
    """
    if compression_level == 0:
        raise ValueError("store_l0 does not have embeddings - use text search")

    if compression_level is not None:
        index = self._index_for_level(compression_level)
    else:
        # Search l1 and l2 (both have embeddings)
        index = "store_l1,store_l2"

    response = await self._client.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": embedding,
            "k": k,
            "num_candidates": num_candidates,
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        record = self.record_class.model_validate(hit["_source"])
        score = hit["_score"]
        results.append((record, score))

    return results
```

## Complete Example

```python
"""
Complete example showing tiered document storage with compression-level routing.
"""

from uuid import uuid4
from core.stores.elasticsearch import ElasticsearchStores
from core.stores.schema import StoreRecord, SourceType
from core.embedding import EmbeddingService

async def process_document(content: str, title: str):
    """Process a document through all compression levels."""
    stores = ElasticsearchStores()
    embedding_service = EmbeddingService()

    try:
        # L0: Store original document (no embedding)
        l0_record = StoreRecord(
            id=uuid4(),
            source_type=SourceType.DOCUMENT,
            content=content,
            compression_level=0,
            metadata={"title": title},
        )
        await stores.store.add(l0_record)  # Routes to store_l0
        print(f"Stored L0: {l0_record.id}")

        # L1: Generate and store short summary (with embedding)
        short_summary = await generate_summary(content, max_words=500)
        l1_embedding = await embedding_service.embed(short_summary)

        l1_record = StoreRecord(
            id=uuid4(),
            source_type=SourceType.INTERNAL,
            content=short_summary,
            compression_level=1,
            source_ids=[l0_record.id],  # Link to original
            embedding=l1_embedding,
            embedding_model=embedding_service.model,
        )
        await stores.store.add(l1_record)  # Routes to store_l1
        print(f"Stored L1: {l1_record.id}")

        # L2: Generate and store 10:1 compressed summary (with embedding)
        tenth_summary = await generate_summary(content, compression_ratio=10)
        l2_embedding = await embedding_service.embed(tenth_summary)

        l2_record = StoreRecord(
            id=uuid4(),
            source_type=SourceType.INTERNAL,
            content=tenth_summary,
            compression_level=2,
            source_ids=[l0_record.id],
            embedding=l2_embedding,
            embedding_model=embedding_service.model,
        )
        await stores.store.add(l2_record)  # Routes to store_l2
        print(f"Stored L2: {l2_record.id}")

        # Search examples
        # Text search across all levels
        results = await stores.store.search(
            {"match": {"content": "key concept"}},
            compression_level=None,  # Uses store_l*
        )

        # Semantic search on summaries only
        query_embedding = await embedding_service.embed("find related concepts")
        semantic_results = await stores.store.knn_search(
            embedding=query_embedding,
            k=5,
            compression_level=None,  # Uses store_l1,store_l2
        )

        return l0_record.id, l1_record.id, l2_record.id

    finally:
        await stores.close()
```

## Consequences

### Benefits

- **Storage efficiency**: Embeddings only stored for summaries, saving ~6KB per original document
- **Schema optimization**: L0 index optimized for text search, L1/L2 for vector search
- **Query safety**: Explicit error for invalid L0 vector searches prevents runtime failures
- **Flexible search**: Wildcard patterns enable cross-level text search while explicit indices optimize vector search
- **Clear data lineage**: `source_ids` field tracks which summaries derive from which originals

### Trade-offs

- **Index proliferation**: 3 indices instead of 1 (more templates to maintain)
- **Consistency window**: During processing, intermediate states may have partial records
- **Embedding dimension coupling**: Hardcoded 1536 dimensions ties to specific embedding model
- **Routing logic**: Callers must understand compression levels for optimal queries

### Alternatives

- **Single index with sparse vectors**: Rejected due to HNSW overhead for null embeddings
- **Separate store classes**: Rejected as it would leak routing logic to callers
- **On-demand summarization**: Rejected due to LLM cost and latency for each search

## Related Patterns

- [MCP Server Store Exposure](./mcp-server-store-exposure.md) - Uses this store for document retrieval
- [Mandatory Archive Before Delete](./mandatory-archive-before-delete.md) - Deletion archival uses same routing
- [GPU-Accelerated Document Processing](../data-pipeline/gpu-accelerated-document-processing.md) - Creates the L0 content

## Known Uses in Thala

- `core/stores/elasticsearch.py`: MainStore class with full routing implementation
- `core/stores/templates/store_l{0,1,2}.json`: Index templates
- `core/stores/setup_indices.py`: Index management CLI
- `workflows/document_processing/nodes/save_short_summary.py`: Creates L1 records
- `workflows/document_processing/nodes/save_tenth_summary.py`: Creates L2 records
- `workflows/document_processing/nodes/store_updater.py`: Updates L0 records

## References

- [Elasticsearch Dense Vector Field](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
- [Elasticsearch KNN Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)
- [Index Templates](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-templates.html)
