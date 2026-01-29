---
name: mandatory-archive-before-delete
title: "Mandatory Archive Before Delete Pattern"
date: 2025-12-17
category: stores
applicability:
  - "When data has legal, regulatory, or business requirements for audit trails"
  - "When users need ability to recover previous states"
  - "When accidental deletion would cause significant harm"
components: [elasticsearch, chroma]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [data-integrity, audit-trail, soft-delete, history-tracking, archiving]
---

# Mandatory Archive Before Delete Pattern

## Intent

Enforce complete state preservation before any destructive operation (update/delete) by making archiving mandatory at the code level, not a bypassable policy.

## Motivation

In systems where data integrity and auditability are critical, accidental or intentional data loss can be catastrophic. This pattern makes history preservation a non-negotiable requirement enforced at runtime. If the archive store is not configured, destructive operations fail immediately with `RuntimeError`, preventing silent data loss from misconfiguration.

## Applicability

Use this pattern when:
- Data has legal, regulatory, or business requirements for audit trails
- Users need ability to recover previous states
- System requires explainability for why data changed
- Multiple stores exist and cross-store archiving is needed

Do NOT use this pattern when:
- High-frequency updates where archive storage would be prohibitive
- Temporary/cache data with no audit requirements
- Performance-critical paths where archive latency is unacceptable

## Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Destructive Operation                     │
│                  (update/delete request)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 1. Check Archive Configured                  │
│         if es_stores is None: raise RuntimeError            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 2. Retrieve Existing State                   │
│              existing = await self.get(record_id)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 3. Archive to History Store                  │
│    WhoIWasRecord(previous_data=snapshot, reason=reason)     │
│              await es_stores.who_i_was.add(...)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 4. Execute Destructive Op                    │
│              await collection.delete(ids=[...])             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Inject Archive Store Dependency

Stores that perform destructive operations receive a reference to the archive store system at construction time:

```python
class ChromaStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "knowledge",
        es_stores: Optional["ElasticsearchStores"] = None,
    ):
        self.collection_name = collection_name
        self._client = chromadb.HttpClient(...)
        self._es_stores = es_stores  # Archive dependency
```

### Step 2: Fail-Fast Guard on Destructive Operations

Every update/delete method begins with a mandatory check:

```python
async def delete(
    self,
    record_id: UUID,
    reason: str = "Deleted via API",
) -> bool:
    """Delete a record with mandatory history tracking."""
    if self._es_stores is None:
        raise RuntimeError(
            "ChromaStore.delete() requires es_stores for mandatory archiving."
        )
    # ... proceed with archive-then-delete ...
```

### Step 3: Archive-Before-Modify Sequence

The pattern enforces a strict sequence: **retrieve -> archive -> modify**

```python
async def delete(self, record_id: UUID, reason: str = "Deleted via API") -> bool:
    # Guard: archiving must be possible
    if self._es_stores is None:
        raise RuntimeError(
            "ChromaStore.delete() requires es_stores for mandatory archiving."
        )

    # STEP 1: Retrieve existing state
    existing = await self.get(record_id)
    if existing is None:
        return False

    # STEP 2: Archive to history store (mandatory)
    who_i_was = WhoIWasRecord(
        supersedes=record_id,
        reason=reason,
        previous_data={
            "metadata": existing["metadata"],
            "document": existing["document"],
        },
        original_store="top_of_mind",
    )
    await self._es_stores.who_i_was.add(who_i_was)

    # STEP 3: Perform the actual deletion
    collection = await self._get_collection()
    await asyncio.to_thread(collection.delete, ids=[str(record_id)])
    return True
```

## Complete Example

```python
"""
Complete working example from core/stores/chroma.py
"""
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from .schema import BaseRecord, WhoIWasRecord

if TYPE_CHECKING:
    from .elasticsearch import ElasticsearchStores


class ChromaStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "knowledge",
        es_stores: Optional["ElasticsearchStores"] = None,
    ):
        self.collection_name = collection_name
        self._es_stores = es_stores

    async def update(
        self,
        record: BaseRecord,
        embedding: list[float],
        document: str,
        reason: str = "Updated via API",
    ) -> UUID:
        """Update a record with history tracking."""
        if self._es_stores is None:
            raise RuntimeError(
                "ChromaStore.update() requires es_stores for mandatory archiving. "
                "Use add() for new records that don't require history tracking."
            )

        existing = await self.get(record.id)

        if existing is not None:
            who_i_was = WhoIWasRecord(
                supersedes=record.id,
                reason=reason,
                previous_data={
                    "metadata": existing["metadata"],
                    "document": existing["document"],
                },
                original_store="top_of_mind",
            )
            await self._es_stores.who_i_was.add(who_i_was)

        return await self.add(record, embedding, document)

    async def delete(
        self,
        record_id: UUID,
        reason: str = "Deleted via API",
    ) -> bool:
        """Delete a record with mandatory history tracking."""
        if self._es_stores is None:
            raise RuntimeError(
                "ChromaStore.delete() requires es_stores for mandatory archiving."
            )

        existing = await self.get(record_id)
        if existing is None:
            return False

        who_i_was = WhoIWasRecord(
            supersedes=record_id,
            reason=reason,
            previous_data={
                "metadata": existing["metadata"],
                "document": existing["document"],
            },
            original_store="top_of_mind",
        )
        await self._es_stores.who_i_was.add(who_i_was)

        collection = await self._get_collection()
        await asyncio.to_thread(collection.delete, ids=[str(record_id)])
        return True
```

## Consequences

### Benefits

- **Complete Audit Trail**: Every change is recorded with reason, timestamp, and full previous state
- **State Recovery**: Can reconstruct any previous state from archived snapshots
- **Fail-Fast Design**: Misconfiguration caught immediately via RuntimeError, not silent data loss
- **Explicit Dependencies**: Cross-store relationships are visible in constructor signatures
- **Semantic Clarity**: Different archive destinations (`who_i_was` vs `forgotten`) convey intent

### Trade-offs

- **Storage Growth**: Archive stores accumulate data; implement retention policies as needed
- **Write Latency**: Archive write adds ~1 round-trip; acceptable for most CRUD operations
- **Coupling**: Stores depend on archive infrastructure; use graceful degradation in tests

### Alternatives

- **Event Sourcing**: Store all changes as events; this pattern is a lighter-weight alternative
- **Soft Delete**: Mark records as deleted without archiving full state; less complete but simpler

## Related Patterns

- **Dependency Injection**: Used for providing archive store references
- **Repository Pattern**: The store classes act as repositories with built-in archiving

## Known Uses in Thala

- `core/stores/chroma.py`: ChromaStore.update() and delete() archive to who_i_was
- `core/stores/elasticsearch.py`: CoherenceStore.delete() archives to who_i_was
- `core/stores/elasticsearch.py`: MainStore.delete() archives to forgotten_store

## References

- Commit c2db0b5: "Add mandatory history tracking for store updates and deletes"
