---
name: two-stage-pipeline-architecture
title: "Two-Stage Pipeline Architecture: Decoupling GPU from IO Workloads"
date: 2026-01-28
category: data-pipeline
applicability:
  - "Pipelines with GPU-bound and IO-bound stages"
  - "PDF processing followed by LLM batch API calls"
  - "Document processing with different resource bottlenecks"
  - "Systems needing memory-efficient queue sizing"
components: [bounded_marker_queue, unbounded_llm_queue, concurrent_workers, batch_collection]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [pipeline, gpu, io-bound, queue-sizing, batch-processing, marker, two-stage, concurrency]
---

# Two-Stage Pipeline Architecture: Decoupling GPU from IO Workloads

## Intent

Separate GPU-bound operations (PDF→markdown conversion) from IO-bound operations (LLM batch API calls) using two queues with different sizing strategies, maximizing GPU utilization during API polling delays.

## Motivation

Single-stage pipelines with mixed GPU/IO workloads waste resources:

**The Problem:**
```
Single-Stage Pipeline (inefficient):
┌─────────────────────────────────────────────────────────────────────┐
│  acquisition → marker_queue → marker+LLM_consumer                   │
│                 (bounded: 8)   (single stage)                       │
│                                                                     │
│  Timeline:                                                          │
│  [0s]   PDF 1 acquired                                             │
│  [2s]   PDF 1 converted (GPU)                                      │
│  [2-7s] PDF 1 LLM processing (GPU IDLE - waiting for batch API)    │
│  [8s]   PDF 2 acquired                                             │
│  [10s]  PDF 2 converted (GPU)                                      │
│                                                                     │
│  GPU Utilization: ~30% (idle during IO-bound LLM calls)            │
└─────────────────────────────────────────────────────────────────────┘
```

**The Solution:**
```
Two-Stage Pipeline (efficient):
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 (GPU-bound):                                               │
│  acquisition → marker_queue → marker_consumer                       │
│                 (bounded: 8)   (4 workers)                          │
│                      ↓                                              │
│  Stage 2 (IO-bound):                                                │
│  llm_queue → llm_consumer                                          │
│  (unbounded)  (4 workers, batched)                                  │
│                                                                     │
│  Timeline:                                                          │
│  [0s]   PDF 1 acquired                                             │
│  [2s]   PDF 1 converted → pushed to llm_queue                      │
│  [2s]   GPU starts PDF 2 conversion (while LLM processes PDF 1)    │
│  [4s]   PDF 2 converted → pushed to llm_queue                      │
│  [4s]   GPU starts PDF 3 conversion                                │
│  [7s]   PDF 1 LLM complete (was processing in background)          │
│                                                                     │
│  GPU Utilization: ~85% (busy during IO-bound LLM calls)            │
└─────────────────────────────────────────────────────────────────────┘
```

## Applicability

Use this pattern when:
- Pipeline has distinct GPU-bound and IO-bound stages
- IO-bound stage has significant latency (API polling, network calls)
- GPU is expensive and should be maximally utilized
- Different stages have different memory footprints
- Need to tune concurrency independently per stage

Do NOT use this pattern when:
- All stages are CPU-bound with similar characteristics
- No significant latency between stages
- Memory is not a concern (can use single unbounded queue)
- Simplicity is more important than efficiency

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  acquisition_producer (concurrent with semaphore)                  │
│  - Downloads PDFs via OA/retrieve-academic                         │
│  - Pushes (doi, pdf_path, metadata) to marker_queue               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  marker_queue (BOUNDED: maxsize=8)                                 │
│  ~400MB max (8 PDFs × 50MB each)                                   │
│  Purpose: Limit memory pressure from large PDFs                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  marker_consumer (Stage 1: GPU-bound)                              │
│  - 4 concurrent workers (MAX_MARKER_CONCURRENT)                    │
│  - PDF → markdown conversion via Marker                            │
│  - Passthrough for already-markdown content                        │
│  - Pushes (doi, markdown, metadata) to llm_queue                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  llm_queue (UNBOUNDED)                                             │
│  ~10MB typical (100 markdowns × 100KB each)                        │
│  Purpose: Keep GPU busy during LLM API polling delays              │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  llm_consumer (Stage 2: IO-bound)                                  │
│  - 4 concurrent workers (MAX_LLM_CONCURRENT)                       │
│  - Batched LLM processing via Anthropic Batch API                  │
│  - Concurrent Zotero/Elasticsearch writes                          │
│  - Returns (doi, result) to caller                                │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Define Queue Configuration

```python
# workflows/research/academic_lit_review/paper_processor/acquisition/types.py

# Stage 1: GPU-bound marker processing
MAX_MARKER_CONCURRENT = 4      # Concurrent marker workers
MARKER_QUEUE_SIZE = 8          # ~400MB max (8 PDFs × 50MB)

# Stage 2: IO-bound LLM processing
MAX_LLM_CONCURRENT = 4         # Concurrent LLM workers
# LLM queue is unbounded: markdown is small (~100KB-1MB vs 50MB PDFs)
```

### Step 2: Create Two-Stage Queue Structure

```python
# workflows/research/academic_lit_review/paper_processor/acquisition/core.py

async def run_paper_pipeline(
    papers: list[PaperMetadata],
    use_batch_api: bool = True,
) -> dict[str, Any]:
    """Run two-stage paper processing pipeline."""

    # Stage 1 queue: Bounded for memory control (large PDFs)
    marker_queue: asyncio.Queue[tuple[str, str, PaperMetadata, bool] | None] = (
        asyncio.Queue(maxsize=MARKER_QUEUE_SIZE)
    )

    # Stage 2 queue: Unbounded (markdown is small, keep GPU busy)
    llm_queue: asyncio.Queue[tuple[str, str, PaperMetadata] | None] = (
        asyncio.Queue()  # No maxsize = unbounded
    )

    # Shared state
    processing_results: dict[str, dict] = {}
    processing_failed: list[str] = []

    # Run all stages concurrently
    await asyncio.gather(
        acquisition_producer(papers, marker_queue),
        marker_consumer(marker_queue, llm_queue),
        llm_consumer(llm_queue, processing_results, processing_failed),
    )

    return {
        "results": processing_results,
        "failed": processing_failed,
    }
```

### Step 3: Implement Stage 1 (GPU-bound Marker)

```python
async def marker_consumer(
    marker_queue: asyncio.Queue,
    llm_queue: asyncio.Queue,
):
    """Stage 1: Convert PDFs to markdown (GPU-bound)."""

    async def marker_worker(worker_id: int):
        """Single marker worker."""
        while True:
            item = await marker_queue.get()
            if item is None:  # Shutdown signal
                await marker_queue.put(None)  # Pass to other workers
                break

            doi, source_path, paper, is_markdown = item

            try:
                if is_markdown:
                    # Already markdown, passthrough
                    markdown = Path(source_path).read_text()
                else:
                    # GPU-bound PDF conversion
                    markdown = await convert_pdf_to_markdown(source_path)

                # Push to Stage 2 (unbounded queue, won't block)
                await llm_queue.put((doi, markdown, paper))

            except Exception as e:
                logger.error(f"Marker failed for {doi}: {e}")
                # Record failure for fallback handling

            finally:
                marker_queue.task_done()

    # Launch concurrent workers
    workers = [marker_worker(i) for i in range(MAX_MARKER_CONCURRENT)]
    await asyncio.gather(*workers)

    # Signal completion to Stage 2
    await llm_queue.put(None)
```

### Step 4: Implement Stage 2 (IO-bound LLM with Batch Collection)

```python
async def llm_consumer(
    llm_queue: asyncio.Queue,
    processing_results: dict[str, dict],
    processing_failed: list[str],
    use_batch_api: bool = True,
):
    """Stage 2: Run LLM workflow using batched processing."""

    async def llm_worker(worker_id: int):
        """Single LLM worker with batch collection."""
        while True:
            # Wait for first item (blocks until available)
            item = await llm_queue.get()
            if item is None:
                await llm_queue.put(None)  # Pass shutdown signal
                break

            # Collect batch: first item + drain available
            batch = [item]
            while True:
                try:
                    next_item = llm_queue.get_nowait()  # Non-blocking
                    if next_item is None:
                        await llm_queue.put(None)
                        break
                    batch.append(next_item)
                except asyncio.QueueEmpty:
                    break

            # Process batch
            if use_batch_api and len(batch) > 1:
                results = await process_multiple_documents(batch)
            else:
                results = []
                for doi, markdown, paper in batch:
                    result = await process_single_document(doi, markdown, paper)
                    results.append(result)

            # Record results
            for (doi, _, _), result in zip(batch, results):
                if result.get("success"):
                    processing_results[doi] = result
                else:
                    processing_failed.append(doi)

    # Launch concurrent workers
    workers = [llm_worker(i) for i in range(MAX_LLM_CONCURRENT)]
    await asyncio.gather(*workers)
```

### Step 5: Implement Batched Document Processing

```python
# workflows/research/academic_lit_review/paper_processor/document_processing.py

async def process_multiple_documents(
    documents: list[tuple[str, str, PaperMetadata]],
    use_batch_api: bool = True,
) -> list[dict[str, Any]]:
    """Process multiple documents using centralized batch processing."""

    # Transform to batch format
    doc_configs = []
    for doi, markdown_text, paper in documents:
        doc_configs.append({
            "source": markdown_text,
            "title": paper.get("title", "Unknown"),
            "item_type": "journalArticle",
            "extra_metadata": {
                "doi": doi,
                "abstract": paper.get("abstract"),
            },
            "use_batch_api": use_batch_api,
        })

    # Use centralized batch processing
    raw_results = await process_documents_batch(doc_configs)

    # Transform back to pipeline format
    results = []
    for (doi, _, paper), raw in zip(documents, raw_results):
        results.append({
            "doi": doi,
            "success": raw.get("current_status") not in ("failed",),
            "es_record_id": raw.get("store_records", [{}])[0].get("id"),
            "zotero_key": raw.get("zotero_key"),
            "validation_status": raw.get("validation_status"),
        })

    return results
```

## Complete Example

```python
import asyncio
from typing import Any

# Configuration
MAX_MARKER_CONCURRENT = 4
MARKER_QUEUE_SIZE = 8
MAX_LLM_CONCURRENT = 4


async def run_two_stage_pipeline(
    papers: list[dict],
    use_batch_api: bool = True,
) -> dict[str, Any]:
    """Complete two-stage pipeline implementation."""

    # Create queues
    marker_queue = asyncio.Queue(maxsize=MARKER_QUEUE_SIZE)  # Bounded
    llm_queue = asyncio.Queue()  # Unbounded

    results = {}
    failed = []

    # Define stages
    async def acquisition_producer():
        for paper in papers:
            pdf_path = await download_pdf(paper["doi"])
            await marker_queue.put((paper["doi"], pdf_path, paper, False))
        await marker_queue.put(None)  # Signal completion

    async def marker_consumer():
        async def worker(i):
            while True:
                item = await marker_queue.get()
                if item is None:
                    await marker_queue.put(None)
                    break
                doi, path, paper, is_md = item
                try:
                    markdown = await convert_pdf(path) if not is_md else Path(path).read_text()
                    await llm_queue.put((doi, markdown, paper))
                finally:
                    marker_queue.task_done()

        await asyncio.gather(*[worker(i) for i in range(MAX_MARKER_CONCURRENT)])
        await llm_queue.put(None)

    async def llm_consumer():
        async def worker(i):
            while True:
                item = await llm_queue.get()
                if item is None:
                    await llm_queue.put(None)
                    break

                # Batch collection
                batch = [item]
                while True:
                    try:
                        next_item = llm_queue.get_nowait()
                        if next_item is None:
                            await llm_queue.put(None)
                            break
                        batch.append(next_item)
                    except asyncio.QueueEmpty:
                        break

                # Process batch
                batch_results = await process_batch(batch, use_batch_api)
                for (doi, _, _), result in zip(batch, batch_results):
                    if result["success"]:
                        results[doi] = result
                    else:
                        failed.append(doi)

        await asyncio.gather(*[worker(i) for i in range(MAX_LLM_CONCURRENT)])

    # Run all stages
    await asyncio.gather(
        acquisition_producer(),
        marker_consumer(),
        llm_consumer(),
    )

    return {"results": results, "failed": failed}


# Usage
result = await run_two_stage_pipeline(papers, use_batch_api=True)
print(f"Processed: {len(result['results'])}, Failed: {len(result['failed'])}")
```

## Consequences

### Benefits

- **Maximized GPU utilization**: GPU converts PDFs while IO waits for batch API
- **Memory efficient**: Different queue sizes match data characteristics
- **Independent tuning**: Adjust marker vs LLM concurrency separately
- **Earlier results**: First paper ready after `marker_time + llm_time` (not `all_marker_times`)
- **Better error isolation**: Failures at marker stage don't affect LLM batches
- **Flexible batching**: LLM workers collect batches dynamically

### Trade-offs

- **Increased complexity**: Two queues, two consumer implementations
- **More coordination**: Shutdown signaling between stages
- **Memory for markdown**: Unbounded LLM queue can grow (but markdown is small)
- **Debugging difficulty**: Concurrent stages harder to trace

### Alternatives

- **Single-stage pipeline**: Simpler but wastes GPU during IO waits
- **Process pools**: Separate processes for GPU/IO (higher overhead)
- **Async generators**: Streaming approach (less batching control)

## Related Patterns

- [Phased Pipeline Architecture for GPU Queue](./phased-pipeline-architecture-gpu-queue.md) - Three-phase pipeline with rate limiting
- [GPU-Accelerated Document Processing](./gpu-accelerated-document-processing.md) - Marker service integration
- [Streaming Async Results Pipeline](../../solutions/async-issues/streaming-async-results-pipeline.md) - Push to queue inside async task

## Known Uses in Thala

- `workflows/research/academic_lit_review/paper_processor/acquisition/core.py` - Main two-stage implementation
- `workflows/research/academic_lit_review/paper_processor/acquisition/types.py` - Queue configuration constants
- `workflows/research/academic_lit_review/paper_processor/document_processing.py` - Batched document processing

## References

- [asyncio.Queue](https://docs.python.org/3/library/asyncio-queue.html)
- [Producer-Consumer Pattern](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem)
- [GPU Utilization Best Practices](https://developer.nvidia.com/blog/cuda-pro-tip-minimize-the-tail-effect/)
