# Stores

Data persistence layer providing unified interfaces for vector storage, structured search, and citation management. All stores share a common UUID-based schema with automatic versioning and archival.

## Usage

### Elasticsearch Stores

Structured storage with multi-level compression and automatic history tracking.

```python
from core.stores import ElasticsearchStores

async with ElasticsearchStores() as stores:
    # Main store - originals and compressions
    record = StoreRecord(
        source_type=SourceType.EXTERNAL,
        zotero_key="ABC12345",
        content="Document text...",
        compression_level=0,
    )
    await stores.store.add(record)

    # Search by zotero_key
    results = await stores.store.search(
        query={"term": {"zotero_key": "ABC12345"}},
        compression_level=0,
    )

    # KNN vector search (L1/L2 only)
    matches = await stores.store.knn_search(
        embedding=[0.1, 0.2, ...],
        k=10,
        compression_level=1,
    )

    # Coherence - identity, beliefs, preferences
    belief = CoherenceRecord(
        content="I prefer functional programming",
        confidence=0.85,
        category="preference",
    )
    await stores.coherence.add(belief)

    # Update creates WhoIWasRecord automatically
    await stores.coherence.update(
        belief.id,
        {"confidence": 0.90, "_change_reason": "New evidence"}
    )

    # Delete with archival
    await stores.store.delete(record.id, reason="Outdated information")
```

### ChromaDB Vector Store

Fast vector retrieval with mandatory history tracking on mutations.

```python
from core.stores import ChromaStore, ElasticsearchStores

es_stores = ElasticsearchStores()
chroma = ChromaStore(es_stores=es_stores)

# Add record with embedding
record = BaseRecord(
    source_type=SourceType.INTERNAL,
    content="Active project note",
)
await chroma.add(record, embedding=vector, document=record.content)

# Update (requires es_stores for WhoIWasRecord)
await chroma.update(
    record,
    new_embedding,
    "Updated content",
    reason="Project status change"
)

# Vector search with metadata filters
results = await chroma.search(
    query_embedding=vector,
    n_results=10,
    where={"category": "project"},
)
```

### Zotero Integration

Local CRUD operations via zotero-local-crud plugin.

```python
from core.stores import ZoteroStore, ZoteroItemCreate, ZoteroCreator

async with ZoteroStore() as zotero:
    # Create citation
    key = await zotero.add(ZoteroItemCreate(
        itemType="journalArticle",
        fields={
            "title": "Paper Title",
            "date": "2024",
            "DOI": "10.1234/example",
        },
        creators=[
            ZoteroCreator(
                firstName="John",
                lastName="Doe",
                creatorType="author",
            )
        ],
        tags=["ai", "research"],
    ))

    # Retrieve item
    item = await zotero.get(key)

    # Search
    results = await zotero.quicksearch("machine learning")
    by_tag = await zotero.search_by_tag("ai")

    # Link to BaseRecord
    record = await zotero.link_record(record, zotero_key=key)
```

### Translation Server

Extract bibliographic metadata from URLs and identifiers.

```python
from core.stores import TranslationServerClient

async with TranslationServerClient() as client:
    # URL translation
    result = await client.translate_url("https://arxiv.org/abs/2301.12345")
    if result:
        print(f"Title: {result.title}")
        print(f"Authors: {[c.to_full_name() for c in result.creators]}")
        print(f"DOI: {result.doi}")

    # Identifier lookup
    result = await client.search_identifier("10.1234/example")
```

### Academic Retrieval

Full-text document retrieval via VPN-enabled service.

```python
from core.stores import RetrieveAcademicClient

async with RetrieveAcademicClient() as client:
    # Check VPN status
    if not await client.health_check():
        raise RuntimeError("VPN required")

    # Submit retrieval job
    job = await client.retrieve(
        doi="10.1234/example",
        title="Example Paper",
        authors=["John Doe"],
    )

    # Wait and download
    result = await client.wait_for_completion(job.job_id, timeout=120)
    if result.status == "completed":
        path = await client.download_file(job.job_id, "/tmp/paper.pdf")

    # Convenience method
    path, result = await client.retrieve_and_download(
        doi="10.1234/example",
        local_path="/tmp/paper.pdf",
    )
```

## Input/Output

### Record Types

| Type | Store | Purpose |
|------|-------|---------|
| `StoreRecord` | Elasticsearch (store_l0/l1/l2) | Main knowledge base with compression levels |
| `CoherenceRecord` | Elasticsearch (coherence) | Identity, beliefs, preferences with confidence |
| `WhoIWasRecord` | Elasticsearch (who_i_was) | Edit history with full snapshots |
| `ForgottenRecord` | Elasticsearch (forgotten) | Archived content with deletion reason |
| `BaseRecord` | Chroma (top_of_mind) | Active projects, fast vector retrieval |

### Common Fields

All records inherit from `BaseRecord`:

```python
id: UUID                          # Primary key
source_type: SourceType           # EXTERNAL (has zotero_key) or INTERNAL
zotero_key: Optional[str]         # 8-char Zotero citation key
content: str                      # Main text for embedding
language_code: Optional[str]      # ISO 639-1 (e.g., "en", "es")
metadata: dict                    # Flexible metadata
embedding: Optional[list[float]]  # Vector for semantic search
created_at: datetime              # Timestamp (UTC)
updated_at: datetime              # Timestamp (UTC)
```

## Architecture

### Multi-Instance Elasticsearch

Separate ES instances for different data lifecycles:

- **Port 9201** (coherence): `store_l0`, `store_l1`, `store_l2`, `coherence`
- **Port 9200** (forgotten): `who_i_was`, `forgotten`

### Compression Levels

MainStore routes to different indices by `compression_level`:

- **L0** (`store_l0`): Original documents, no embeddings
- **L1** (`store_l1`): Short summaries with embeddings
- **L2** (`store_l2`): 10:1 compressions with embeddings

### Automatic Versioning

CoherenceStore and ChromaStore automatically create `WhoIWasRecord` entries on update/delete, preserving full snapshots in `previous_data` field.

### Cross-Store Verification

Utility functions verify data consistency across Zotero and Elasticsearch:

```python
from core.stores import verify_zotero_keys, verify_zotero_keys_batch

# Verify keys exist in both stores
results = await verify_zotero_keys(["ABC12345", "DEF67890"])
for result in results:
    print(f"{result.zotero_key}: Zotero={result.exists_in_zotero}, ES={result.es_record_id}")

# Batch lookup returning dict
lookup = await verify_zotero_keys_batch({"ABC12345", "DEF67890"})
if lookup["ABC12345"].exists_in_zotero:
    print("Found in Zotero")
```

## Configuration

### Environment Variables

```bash
# Elasticsearch
THALA_ES_COHERENCE_HOST=http://localhost:9201
THALA_ES_FORGOTTEN_HOST=http://localhost:9200

# ChromaDB
THALA_CHROMA_HOST=localhost
THALA_CHROMA_PORT=8000

# Translation Server
THALA_TRANSLATION_HOST=localhost
THALA_TRANSLATION_PORT=1969

# Academic Retrieval
THALA_RETRIEVE_ACADEMIC_HOST=localhost
THALA_RETRIEVE_ACADEMIC_PORT=8002
```

### Index Setup

Run once during initial setup or after schema changes:

```bash
# Create indices with templates
python -m core.stores.setup_indices

# Reset indices (delete and recreate)
python -m core.stores.setup_indices --reset

# Show index status and document counts
python -m core.stores.setup_indices --status
```

## Related Modules

- [core/config.py](../config.py) - Environment configuration
- [core/embedding.py](../embedding.py) - Embedding service for vector generation
- [core/scraping/](../scraping/) - Web scraping for external content
- [workflows/shared/](../../workflows/shared/) - Caching and LLM utilities
