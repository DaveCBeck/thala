---
name: two-stage-pipeline-architecture
title: "Two-Stage Pipeline Architecture: Decoupling GPU from IO Workloads"
date: 2026-01-28
category: data-pipeline
applicability:
  - "Pipelines with GPU-bound and IO-bound stages"
  - "PDF processing followed by LLM batch API calls"
  - "Document processing with different resource bottlenecks"
  - "Systems needing to maximize GPU utilization"
components: [marker_queue, llm_queue, smart_routing, batch_collection]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [pipeline, gpu, io-bound, queue-sizing, batch-processing, marker, two-stage, concurrency, smart-routing]
---

# Two-Stage Pipeline Architecture: Decoupling GPU from IO Workloads

## Intent

Separate GPU-bound operations (PDF->markdown conversion) from IO-bound operations (LLM batch API calls) using two queues with unbounded sizing, maximizing GPU utilization during API polling delays while leveraging smart routing to send simple PDFs to a fast CPU path.

## Motivation

Single-stage pipelines with mixed GPU/IO workloads waste resources:

**The Problem:**
```
Single-Stage Pipeline (inefficient):
+---------------------------------------------------------------------+
|  acquisition -> marker_queue -> marker+LLM_consumer                  |
|                                 (single stage)                       |
|                                                                      |
|  Timeline:                                                           |
|  [0s]   PDF 1 acquired                                              |
|  [2s]   PDF 1 converted (GPU)                                       |
|  [2-7s] PDF 1 LLM processing (GPU IDLE - waiting for batch API)     |
|  [8s]   PDF 2 acquired                                              |
|  [10s]  PDF 2 converted (GPU)                                       |
|                                                                      |
|  GPU Utilization: ~30% (idle during IO-bound LLM calls)             |
+---------------------------------------------------------------------+
```

**The Solution:**
```
Two-Stage Pipeline with Smart Routing (efficient):
+---------------------------------------------------------------------+
|  Stage 1 (Smart Routing):                                            |
|  acquisition -> marker_queue -> marker_consumer                      |
|                 (unbounded)    (smart: CPU or GPU path)              |
|                      |                                               |
|  Stage 2 (IO-bound):                                                 |
|  llm_queue -> llm_consumer                                          |
|  (unbounded)  (batched)                                              |
|                                                                      |
|  Timeline:                                                           |
|  [0s]   PDF 1 acquired                                              |
|  [0.1s] PDF 1 routed to CPU (simple digital doc) -> llm_queue       |
|  [0.1s] PDF 2 acquired                                              |
|  [2s]   PDF 2 routed to GPU (scanned/complex) -> llm_queue          |
|  [2s]   PDF 3 acquired, routed to CPU                               |
|  [5s]   PDF 1 LLM complete (was processing in background)           |
|                                                                      |
|  GPU Utilization: ~90% (only used for complex PDFs)                 |
+---------------------------------------------------------------------+
```

## Applicability

Use this pattern when:
- Pipeline has distinct GPU-bound and IO-bound stages
- IO-bound stage has significant latency (API polling, network calls)
- GPU is expensive and should be reserved for complex documents
- Many documents are simple digital-native PDFs (benefit from CPU path)
- Need to tune concurrency independently per stage

Do NOT use this pattern when:
- All stages are CPU-bound with similar characteristics
- No significant latency between stages
- All documents are complex/scanned (no benefit from smart routing)
- Simplicity is more important than efficiency

## Structure

```
+--------------------------------------------------------------------+
|  acquisition_producer (concurrent with semaphore)                   |
|  - Downloads PDFs via OA/retrieve-academic                          |
|  - Pushes (doi, pdf_path, metadata) to marker_queue                |
+--------------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------------+
|  marker_queue (UNBOUNDED)                                           |
|  Holds file paths (strings) or markdown text, NOT PDF bytes         |
|  Memory: negligible (~1KB per item for paths)                       |
+--------------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------------+
|  marker_consumer (Stage 1: Smart Routing)                           |
|  - Analyzes PDF complexity (PyMuPDF)                                |
|  - Simple PDFs -> CPU path (PyMuPDF text extraction)                |
|  - Complex/scanned PDFs -> GPU path (Marker service)                |
|  - Marker has its own Redis/Celery queue for GPU jobs               |
|  - Pushes (doi, markdown, metadata) to llm_queue                   |
+--------------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------------+
|  llm_queue (UNBOUNDED)                                              |
|  ~10MB typical (100 markdowns x 100KB each)                         |
|  Purpose: Keep processing flowing during LLM API polling delays     |
+--------------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------------+
|  llm_consumer (Stage 2: IO-bound)                                   |
|  - 4 concurrent workers (MAX_LLM_CONCURRENT)                        |
|  - Batched LLM processing via Anthropic Batch API                   |
|  - Concurrent Zotero/Elasticsearch writes                           |
|  - Returns (doi, result) to caller                                 |
+--------------------------------------------------------------------+
```

## Implementation

### Step 1: Define Queue Configuration

```python
# workflows/research/academic_lit_review/paper_processor/acquisition/types.py

# Two-stage pipeline constants
# Stage 1: Smart routing to CPU (PyMuPDF) or GPU (Marker)
# Marker service has its own Redis/Celery queue that handles GPU job queuing.
# We use unbounded asyncio queues since they hold file paths or markdown text,
# not PDF bytes in memory.

# Stage 2: IO-bound LLM processing
MAX_LLM_CONCURRENT = 4         # Concurrent LLM workers
```

### Step 2: Create Two-Stage Queue Structure

```python
# workflows/research/academic_lit_review/paper_processor/acquisition/core.py

async def run_paper_pipeline(
    papers: list[PaperMetadata],
    use_batch_api: bool = True,
) -> dict[str, Any]:
    """Run two-stage paper processing pipeline."""

    # Both queues unbounded - hold file paths/markdown, not PDF bytes
    marker_queue: asyncio.Queue = asyncio.Queue()
    llm_queue: asyncio.Queue = asyncio.Queue()

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

### Step 3: Implement Stage 1 (Smart Routing)

```python
from core.scraping.pdf import process_document_smart

async def marker_consumer(
    marker_queue: asyncio.Queue,
    llm_queue: asyncio.Queue,
):
    """Stage 1: Convert PDFs via smart routing (CPU or GPU path).

    Uses smart routing to send simple PDFs to fast CPU path (PyMuPDF)
    and complex/scanned PDFs to GPU path (Marker). The Marker service
    has its own Redis/Celery queue for GPU jobs.
    """
    active_tasks: set[asyncio.Task] = set()

    async def process_item(doi, source, paper, is_markdown):
        try:
            if is_markdown:
                # Already markdown, passthrough
                markdown = source
            else:
                # Smart routing: CPU for simple, GPU for complex
                pdf_bytes = Path(source).read_bytes()
                result = await process_document_smart(pdf_bytes)
                markdown = result.markdown
                logger.debug(f"PDF processed via {result.processing_path}: {doi}")

            # Push to Stage 2
            await llm_queue.put((doi, markdown, paper))

        except Exception as e:
            logger.error(f"Processing failed for {doi}: {e}")

    while True:
        item = await marker_queue.get()
        if item is None:  # Shutdown signal
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            await llm_queue.put(None)
            break

        doi, source, paper, is_markdown = item
        task = asyncio.create_task(process_item(doi, source, paper, is_markdown))
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)
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

## Key Design Decisions

### Why Unbounded Queues?

Both queues are unbounded because:

1. **marker_queue holds file paths or markdown text**, not PDF bytes in memory
   - File paths: ~100 bytes each
   - Markdown from OA: ~100KB-1MB each
   - NOT 50MB PDF bytes (those stay on disk)

2. **Marker service has its own Redis/Celery queue**
   - GPU jobs queue inside Marker's infrastructure
   - Celery handles backpressure automatically
   - We don't need client-side semaphores

3. **llm_queue holds markdown text**
   - ~100KB-1MB per document
   - Even 100 items = ~100MB max (acceptable)

### Why No Marker Semaphore?

The previous design had `MAX_MARKER_CONCURRENT = 4` but:

1. Marker uses `--pool=solo` (single-threaded Celery worker)
2. `worker_prefetch_multiplier=1` means 1 task at a time
3. Jobs naturally queue in Redis
4. Client-side semaphore added complexity without benefit

### Smart Routing Benefits

- Simple digital PDFs (70-80% of academic papers): CPU path in ~100ms
- Complex/scanned PDFs: GPU path via Marker, queued appropriately
- Automatic quality selection based on document complexity

## Consequences

### Benefits

- **Maximized GPU utilization**: GPU only processes complex documents
- **Faster throughput**: Simple PDFs skip GPU entirely
- **Simplified code**: No client-side semaphores needed
- **Better resource allocation**: CPU handles what CPU is good at
- **Independent tuning**: Adjust LLM concurrency separately
- **Earlier results**: First paper ready after `smart_route_time + llm_time`

### Trade-offs

- **Increased complexity**: Two queues, smart routing logic
- **More coordination**: Shutdown signaling between stages
- **Analysis overhead**: Each PDF analyzed before routing (~50ms)
- **Debugging difficulty**: Concurrent stages harder to trace

### Alternatives

- **Single-stage pipeline**: Simpler but wastes GPU during IO waits
- **All-GPU pipeline**: Simpler but slow for simple PDFs
- **Process pools**: Separate processes for GPU/IO (higher overhead)

## Related Patterns

- [Phased Pipeline Architecture for GPU Queue](./phased-pipeline-architecture-gpu-queue.md) - Three-phase pipeline with rate limiting
- [GPU-Accelerated Document Processing](./gpu-accelerated-document-processing.md) - Marker service integration
- [Streaming Async Results Pipeline](../../solutions/async-issues/streaming-async-results-pipeline.md) - Push to queue inside async task

## Known Uses in Thala

- `workflows/research/academic_lit_review/paper_processor/acquisition/core.py` - Main two-stage implementation
- `workflows/research/academic_lit_review/paper_processor/acquisition/types.py` - Queue configuration constants
- `workflows/research/academic_lit_review/paper_processor/document_processing.py` - Batched document processing
- `core/scraping/pdf/router.py` - Smart routing implementation

## References

- [asyncio.Queue](https://docs.python.org/3/library/asyncio-queue.html)
- [Producer-Consumer Pattern](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem)
- [GPU Utilization Best Practices](https://developer.nvidia.com/blog/cuda-pro-tip-minimize-the-tail-effect/)
