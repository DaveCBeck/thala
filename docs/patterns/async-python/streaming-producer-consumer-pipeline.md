---
name: streaming-producer-consumer-pipeline
title: Streaming Producer-Consumer Pipeline Pattern
date: 2026-01-03
category: async-python
applicability:
  - "Pipelines with acquisition and processing stages"
  - "Maximizing GPU/resource utilization during I/O operations"
  - "Bounded memory consumption for large batch processing"
  - "Reducing time-to-first-result in multi-stage pipelines"
components: [acquisition, processing_queue, paper_processor]
complexity: medium
verified_in_production: true
tags: [asyncio, producer-consumer, queue, streaming, parallelism, gpu-utilization]
---

# Streaming Producer-Consumer Pipeline Pattern

## Intent

Replace sequential phased pipelines with streaming producer-consumer architecture using `asyncio.Queue` to start processing as soon as the first item is acquired, maximizing resource utilization and reducing time-to-first-result.

## Problem

Sequential phased pipelines have inefficiencies:

```
Traditional Phased Pipeline:
┌─────────────────────────────────────┬────────────────────────────────┐
│   Phase 2: Acquire All PDFs         │   Phase 3: Process All PDFs    │
│   (GPU idle, downloading)           │   (Network idle, processing)   │
└─────────────────────────────────────┴────────────────────────────────┘
Total time = Acquisition time + Processing time
GPU utilization: 0% during acquisition, 100% during processing
```

Issues:
- **GPU idle during acquisition**: Workers wait while all downloads complete
- **Time to first result**: Must wait for slowest download before any processing
- **Memory spike**: All PDFs downloaded before any processing starts
- **No overlap**: I/O-bound and compute-bound phases don't overlap

## Solution

Stream items from acquisition to processing using a bounded `asyncio.Queue`:

```
Streaming Producer-Consumer Pipeline:
┌──────────────────────────────────────────────────────────────────────┐
│ Producer (Acquisition)     Queue       Consumer (Processing)         │
│ ══►  ══►  ══►  ══►  ══►   [▓▓▓░░░░░]  ████████████████████████████  │
│     (rate-limited)         (bounded)   (parallel, fills GPU queue)   │
└──────────────────────────────────────────────────────────────────────┘
Total time ≈ max(Acquisition time, Processing time)
GPU utilization: Near 100% throughout pipeline
```

Benefits:
- Processing starts after first download completes
- GPU stays busy throughout
- Memory bounded by queue size
- I/O and compute overlap

## Structure

```
workflows/research/subgraphs/academic_lit_review/paper_processor/
├── acquisition.py          # Producer-consumer pipeline
│   ├── _check_cache_phase()       # Phase 1: Parallel cache check
│   ├── acquisition_producer()     # Phase 2: Submit jobs, poll, queue
│   └── processing_consumer()      # Phase 3: Process from queue

core/stores/retrieve_academic.py
└── poll_jobs_until_complete()     # Async generator yielding completions
```

## Implementation

### Queue Configuration

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py

# Queue size for streaming pipeline - balances memory vs latency
# ~8 PDFs × 50MB = ~400MB buffer max
PROCESSING_QUEUE_SIZE = 8

# Maximum parallel processing tasks
MAX_PROCESSING_CONCURRENT = 4
```

### Producer (Acquisition)

```python
async def acquisition_producer(
    papers_to_acquire: list[PaperMetadata],
    processing_queue: asyncio.Queue,
    max_concurrent: int,
):
    """Submit jobs with rate limiting, poll completions, push to queue."""
    semaphore = asyncio.Semaphore(max_concurrent)
    acquired_count = 0
    total = len(papers_to_acquire)

    async def submit_single(paper: PaperMetadata, index: int):
        """Submit a single retrieval job with rate limiting."""
        async with semaphore:
            if index > 0:
                await asyncio.sleep(ACQUISITION_DELAY)  # Rate limiting

            doi = paper.get("doi")
            job = await client.retrieve(
                doi=doi,
                title=paper.get("title"),
                authors=[a.get("name") for a in paper.get("authors", [])[:5]],
            )
            return doi, job.job_id, local_path

    try:
        # Submit all jobs with rate limiting
        submit_tasks = [
            submit_single(p, i) for i, p in enumerate(papers_to_acquire)
        ]
        valid_jobs = await asyncio.gather(*submit_tasks, return_exceptions=True)

        # Filter successful submissions
        valid_jobs = [r for r in valid_jobs if not isinstance(r, Exception)]

        logger.info(f"Submitted {len(valid_jobs)} jobs, polling for completions...")

        # Poll and push completions to processing queue
        async for doi, local_path, result in client.poll_jobs_until_complete(
            valid_jobs,
            poll_interval=2.0,
        ):
            if isinstance(result, Exception):
                acquisition_failed.append(doi)
            else:
                acquired_count += 1
                logger.info(f"[{acquired_count}/{total}] Acquired: {doi}")
                # Push to processing queue (blocks if full - backpressure)
                await processing_queue.put((doi, local_path, papers_by_doi[doi]))

    finally:
        # Signal end of acquisitions
        await processing_queue.put(None)
```

### Consumer (Processing)

```python
async def processing_consumer(
    processing_queue: asyncio.Queue,
    processing_results: dict,
    processing_failed: list,
):
    """Process papers from queue as they arrive."""
    process_semaphore = asyncio.Semaphore(MAX_PROCESSING_CONCURRENT)
    active_tasks: set[asyncio.Task] = set()
    processed_count = 0

    async def process_item(doi: str, path: str, paper: PaperMetadata):
        """Process a single document with concurrency limiting."""
        nonlocal processed_count
        async with process_semaphore:
            result = await process_single_document(doi, path, paper)
            processed_count += 1

            if result.get("success"):
                processing_results[doi] = result
                logger.info(f"Processed: {paper.get('title', 'Unknown')[:50]}")
            else:
                processing_failed.append(doi)

    while True:
        item = await processing_queue.get()

        if item is None:
            # End signal - wait for active tasks
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            break

        doi, path, paper = item

        # Create processing task (doesn't wait)
        task = asyncio.create_task(process_item(doi, path, paper))
        active_tasks.add(task)
        task.add_done_callback(active_tasks.discard)
```

### Polling Async Generator

```python
# core/stores/retrieve_academic.py

async def poll_jobs_until_complete(
    self,
    jobs: list[tuple[str, str, str]],  # (doi, job_id, local_path)
    poll_interval: float = 2.0,
    timeout: float = 300.0,
) -> AsyncGenerator[tuple[str, str, Any], None]:
    """Yield results as each job completes.

    Async generator pattern allows streaming results to consumer
    as they become available, rather than waiting for all.
    """
    pending = {job_id: (doi, local_path) for doi, job_id, local_path in jobs}
    start_time = asyncio.get_event_loop().time()

    while pending:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            for job_id, (doi, local_path) in pending.items():
                yield doi, local_path, TimeoutError(f"Job {job_id} timed out")
            break

        # Poll all pending jobs
        for job_id in list(pending.keys()):
            status = await self._get_job_status(job_id)

            if status.state == "completed":
                doi, local_path = pending.pop(job_id)
                # Download the file
                await self._download_result(job_id, local_path)
                yield doi, local_path, status

            elif status.state == "failed":
                doi, local_path = pending.pop(job_id)
                yield doi, local_path, Exception(status.error)

        if pending:
            await asyncio.sleep(poll_interval)
```

### Main Pipeline

```python
async def run_paper_pipeline(papers: list[PaperMetadata]) -> tuple[...]:
    """Run streaming acquire→process pipeline."""
    papers_by_doi = {p.get("doi"): p for p in papers}

    # Phase 1: Check cache (parallel, fast)
    cached_results, papers_to_acquire = await _check_cache_phase(papers)

    if not papers_to_acquire:
        return {}, cached_results, [], []

    # Shared state
    acquired_paths: dict[str, str] = {}
    acquisition_failed: list[str] = []
    processing_results: dict[str, dict] = dict(cached_results)
    processing_failed: list[str] = []

    # Bounded queue for backpressure
    processing_queue: asyncio.Queue = asyncio.Queue(maxsize=PROCESSING_QUEUE_SIZE)

    # Run producer and consumer concurrently
    await asyncio.gather(
        acquisition_producer(papers_to_acquire, processing_queue, ...),
        processing_consumer(processing_queue, processing_results, processing_failed),
    )

    return acquired_paths, processing_results, acquisition_failed, processing_failed
```

## Usage

```python
from workflows.research.subgraphs.academic_lit_review.paper_processor import (
    run_paper_pipeline,
)

# Streaming pipeline: processing starts after first download
acquired, results, acq_failed, proc_failed = await run_paper_pipeline(
    papers=papers_to_process,
    max_concurrent=2,  # Acquisition rate limiting
)

# Results stream in as processing completes
```

## Guidelines

### Queue Size Selection

| Queue Size | Trade-off |
|------------|-----------|
| Small (2-4) | Lower memory, more backpressure |
| Medium (8) | Balanced memory and throughput |
| Large (16+) | Higher memory, smoother flow |

Rule: `queue_size ≈ 2 × processing_concurrency`

### Backpressure Handling

The bounded queue creates natural backpressure:
- If processing is slow, `processing_queue.put()` blocks
- Producer slows acquisition rate automatically
- Memory stays bounded

### End Signal Pattern

Use `None` as sentinel to signal producer completion:
```python
# Producer
finally:
    await processing_queue.put(None)

# Consumer
item = await processing_queue.get()
if item is None:
    break  # Exit loop
```

### Task Tracking

Track active tasks for clean shutdown:
```python
active_tasks: set[asyncio.Task] = set()

task = asyncio.create_task(process_item(...))
active_tasks.add(task)
task.add_done_callback(active_tasks.discard)

# On shutdown
await asyncio.gather(*active_tasks, return_exceptions=True)
```

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py` - Paper pipeline
- `core/stores/retrieve_academic.py` - `poll_jobs_until_complete()` generator

## Consequences

### Benefits
- **Time to first result**: Single acquisition vs sum of all
- **GPU utilization**: Near 100% vs 0% during acquisition
- **Memory bounded**: Queue size limits buffered items
- **Total time**: `max(acquisition, processing)` vs `acquisition + processing`

### Trade-offs
- **Complexity**: Producer-consumer vs simple sequential
- **Error handling**: Must handle errors in both producer and consumer
- **State coordination**: Shared state requires careful design

## Related Patterns

- [Phased Pipeline Architecture](../data-pipeline/phased-pipeline-architecture-gpu-queue.md) - Original phased approach
- [Hash-Based Persistent Caching](../data-pipeline/hash-based-persistent-caching.md) - Cache check phase

## References

- [asyncio.Queue](https://docs.python.org/3/library/asyncio-queue.html)
- [Producer-Consumer Pattern](https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem)
