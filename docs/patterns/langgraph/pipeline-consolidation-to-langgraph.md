---
name: pipeline-consolidation-to-langgraph
title: "Pipeline Consolidation to LangGraph: Eliminating Divergent Implementations"
date: 2026-01-28
category: langgraph
applicability:
  - "Codebases with duplicate batch/single-document processing paths"
  - "Legacy batch implementations with divergent prompt engineering"
  - "Systems needing unified language detection and validation"
  - "Cost optimization through prompt caching vs batch API"
components: [unified_pipeline, semaphore_concurrency, parameter_threading, result_transformation]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [consolidation, langgraph, batch-processing, refactoring, single-source-of-truth, concurrency]
---

# Pipeline Consolidation to LangGraph: Eliminating Divergent Implementations

## Intent

Replace divergent batch and single-document processing implementations with a unified LangGraph pipeline, achieving consistency, better cost optimization via prompt caching, and simpler concurrency control through semaphores.

## Motivation

Codebases often accumulate separate implementations for batch vs single-document processing:

**The Problem:**
```
├── process_document()                    # Single document via LangGraph
│   ├── Language detection ✓
│   ├── Validation ✓
│   ├── Modern prompts ✓
│   └── Prompt caching ✓
│
├── BatchDocumentProcessor                # Batch via separate implementation
│   ├── Language detection ✗ (English only)
│   ├── Validation ✗
│   ├── Different prompts ✗ (17 lines vs 25 lines)
│   └── Prompt caching ✗
│
Results:
- Same document processed differently depending on path
- 462 extra lines to maintain
- Prompt changes require updating two places
- Batch mode misses features added to LangGraph pipeline
```

**The Solution:**
```
├── process_documents_batch(docs, concurrency=5)
│   └── For each doc (semaphore limited):
│       └── process_document()            # Single unified path
│           ├── Language detection ✓
│           ├── Validation ✓
│           ├── Modern prompts ✓
│           └── Prompt caching ✓ (90% savings)
│
Results:
- All documents processed identically
- -462 lines (batch module removed)
- Single source of truth for prompts
- All features available to all processing paths
```

## Applicability

Use this pattern when:
- Batch and single-document paths have diverged
- Batch mode lacks features present in single-document pipeline
- Prompt caching provides better cost savings than batch API
- Need unified behavior (language detection, validation)
- Maintenance burden of dual implementations is unacceptable

Do NOT use this pattern when:
- Batch API provides critical latency-insensitive cost savings
- Truly separate requirements for batch vs single document
- Single-document pipeline cannot handle batch workloads
- Prompt caching is not applicable (each prompt unique)

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  BEFORE: Divergent Paths                                           │
│                                                                    │
│  ┌─────────────────────┐    ┌─────────────────────┐               │
│  │ process_document()  │    │ BatchProcessor      │               │
│  │ (LangGraph)         │    │ (Separate module)   │               │
│  ├─────────────────────┤    ├─────────────────────┤               │
│  │ ✓ Language detect   │    │ ✗ English only      │               │
│  │ ✓ Validation        │    │ ✗ No validation     │               │
│  │ ✓ Modern prompts    │    │ ✗ Old prompts       │               │
│  │ ✓ Prompt caching    │    │ ✗ No caching        │               │
│  └─────────────────────┘    └─────────────────────┘               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  AFTER: Unified Pipeline                                           │
│                                                                    │
│  process_documents_batch(documents, concurrency=5)                │
│      │                                                            │
│      ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────┐│
│  │ Semaphore (concurrency=5)                                     ││
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  ││
│  │ │process_document │ │process_document │ │process_document │  ││
│  │ │ (LangGraph)     │ │ (LangGraph)     │ │ (LangGraph)     │  ││
│  │ │ ✓ Language      │ │ ✓ Language      │ │ ✓ Language      │  ││
│  │ │ ✓ Validation    │ │ ✓ Validation    │ │ ✓ Validation    │  ││
│  │ │ ✓ Caching       │ │ ✓ Caching       │ │ ✓ Caching       │  ││
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘  ││
│  └───────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Create Unified Batch Entry Point

```python
# workflows/document_processing/graph.py

import asyncio
from typing import Any


async def process_documents_batch(
    documents: list[dict[str, Any]],
    concurrency: int = 5,
) -> list[dict[str, Any]]:
    """Process multiple documents through unified LangGraph pipeline.

    Uses semaphore-based concurrency control instead of Anthropic Batch API.
    All documents go through the same pipeline, ensuring consistent behavior.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def process_with_limit(doc_config: dict) -> dict[str, Any]:
        async with semaphore:
            return await process_document(
                source=doc_config["source"],
                title=doc_config.get("title"),
                item_type=doc_config.get("item_type", "document"),
                langs=doc_config.get("langs"),
                extra_metadata=doc_config.get("extra_metadata"),
                use_batch_api=doc_config.get("use_batch_api", True),
            )

    tasks = [process_with_limit(doc) for doc in documents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to error dicts
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "current_status": "failed",
                "errors": [{"type": "exception", "message": str(result)}],
            })
        else:
            final_results.append(result)

    return final_results
```

### Step 2: Thread Parameters Through Pipeline

```python
# workflows/document_processing/graph.py

async def process_document(
    source: str,
    title: str = None,
    item_type: str = "document",
    langs: list[str] = None,
    extra_metadata: dict = None,
    use_batch_api: bool = True,  # Thread through to nodes
) -> dict[str, Any]:
    """Process single document through LangGraph pipeline."""

    initial_state = {
        "input": {
            "source": source,
            "title": title,
            "item_type": item_type,
            "langs": langs or ["English"],
            "extra_metadata": extra_metadata or {},
            "use_batch_api": use_batch_api,  # Available to all nodes
        },
        "current_status": "processing",
        "errors": [],
    }

    result = await workflow_graph.ainvoke(initial_state)
    return result
```

### Step 3: Update Callers to Use Unified Entry Point

```python
# workflows/research/academic_lit_review/paper_processor/document_processing.py

from workflows.document_processing import process_documents_batch


async def process_multiple_documents(
    documents: list[tuple[str, str, PaperMetadata]],
    use_batch_api: bool = True,
) -> list[dict[str, Any]]:
    """Process documents through unified LangGraph pipeline."""

    # Transform to unified config format
    doc_configs = []
    for doi, markdown_text, paper in documents:
        doc_configs.append({
            "source": markdown_text,
            "title": paper.get("title", "Unknown"),
            "item_type": "journalArticle",
            "extra_metadata": {
                "DOI": doi,
                "date": paper.get("publication_date", ""),
                "publicationTitle": paper.get("venue", ""),
                "abstract": paper.get("abstract"),
            },
            "use_batch_api": use_batch_api,
        })

    # Use unified batch processing
    raw_results = await process_documents_batch(doc_configs)

    # Transform to domain format
    results = []
    for (doi, _, _), raw in zip(documents, raw_results):
        store_records = raw.get("store_records", [])
        results.append({
            "doi": doi,
            "success": raw.get("current_status") not in ("failed",),
            "es_record_id": store_records[0].get("id") if store_records else None,
            "zotero_key": raw.get("zotero_key"),
            "short_summary": raw.get("short_summary", ""),
            "original_language": raw.get("original_language", "en"),
            "errors": raw.get("errors", []),
        })

    return results
```

### Step 4: Remove Legacy Batch Module

```python
# workflows/document_processing/__init__.py

# BEFORE
from .batch_mode import (
    BatchDocumentProcessor,
    BatchDocumentRequest,
    BatchDocumentResult,
    process_documents_with_batch_api,
)

# AFTER - Remove all batch_mode imports
from .graph import process_document, process_documents_batch

__all__ = [
    "process_document",
    "process_documents_batch",
]
```

Delete the entire `batch_mode/` directory:
- `batch_mode/__init__.py`
- `batch_mode/job_manager.py`
- `batch_mode/processor.py`
- `batch_mode/types.py`

## Complete Example

```python
from workflows.document_processing import process_documents_batch

# Process multiple documents with unified pipeline
documents = [
    {
        "source": "# Paper 1\n\nContent about machine learning...",
        "title": "Machine Learning Paper",
        "item_type": "journalArticle",
        "extra_metadata": {"DOI": "10.1234/ml.2024"},
        "use_batch_api": True,
    },
    {
        "source": "# Paper 2 (German)\n\nInhalt über künstliche Intelligenz...",
        "title": "German AI Paper",
        "item_type": "journalArticle",
        "extra_metadata": {"DOI": "10.1234/ai.2024"},
    },
]

# All documents go through unified LangGraph pipeline
results = await process_documents_batch(documents, concurrency=5)

for doc, result in zip(documents, results):
    print(f"DOI: {doc['extra_metadata']['DOI']}")
    print(f"  Status: {result['current_status']}")
    print(f"  Language: {result.get('original_language', 'en')}")  # Detected!
    print(f"  Validated: {result.get('validation_passed', True)}")  # Validated!
    print(f"  Summary: {result.get('short_summary', '')[:50]}...")
```

## Consequences

### Benefits

- **Single source of truth**: All documents processed through same pipeline
- **Unified features**: Language detection, validation available to all paths
- **Better cost savings**: Prompt caching (90%) vs batch API (50%)
- **Immediate results**: No async polling delays
- **Simpler maintenance**: Change prompts in one place
- **Reduced code**: -462 lines (batch module removed)
- **Predictable concurrency**: Semaphore provides clear limits

### Trade-offs

- **No async batch savings**: Lose 50% batch API discount (but gain 90% caching)
- **Real-time resources**: Need immediate capacity (vs deferred processing)
- **Breaking change**: Callers must migrate to new API
- **Semaphore tuning**: Must choose appropriate concurrency limit

### Alternatives

- **Keep both paths**: Maintain batch and single-document (higher maintenance)
- **Batch API with caching**: Combine approaches (more complex)
- **Queue-based processing**: External job queue (infrastructure overhead)

## Related Patterns

- [Batch API Cost Optimization](../llm-interaction/batch-api-cost-optimization.md) - When batch API is still needed
- [Anthropic Prompt Caching](../llm-interaction/anthropic-prompt-caching-cost-optimization.md) - The caching that replaces batch savings
- [Workflow Modularization](./workflow-modularization-pattern.md) - How to structure unified pipelines

## Known Uses in Thala

- `workflows/document_processing/graph.py` - Unified `process_documents_batch`
- `workflows/document_processing/__init__.py` - Simplified exports
- `workflows/research/academic_lit_review/paper_processor/document_processing.py` - Caller migration

## References

- [asyncio.Semaphore](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Consolidation Refactoring](https://refactoring.guru/refactoring/techniques/simplifying-method-calls)
