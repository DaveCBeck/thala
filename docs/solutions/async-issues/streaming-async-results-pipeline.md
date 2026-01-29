---
module: workflows/research/academic_lit_review/paper_processor
date: 2026-01-18
problem_type: performance_issue
component: acquisition_core
symptoms:
  - "Marker GPU sits idle during entire OA acquisition phase"
  - "Processing only starts after all downloads complete"
  - "Long delays before first results appear"
  - "Pipeline stages running sequentially instead of overlapped"
root_cause: async_blocking_pattern
resolution_type: code_fix
severity: medium
tags: [async, streaming, pipeline, asyncio-gather, queue, gpu-utilization, throughput]
---

# Streaming Async Results to Pipeline Queue

## Problem

All OA (Open Access) download attempts had to complete before any papers were pushed to the marker processing queue. This caused the GPU-intensive marker stage to sit idle during the entire acquisition phase.

### Symptoms

```
# Timeline showing the blocking pattern
[0.0s]  Start: 4 papers to acquire
[0.5s]  OA #1 downloaded ← Could process NOW
[1.2s]  OA #2 downloaded ← Could process NOW
[5.0s]  OA #3 downloaded
[8.0s]  OA #4 downloaded (slowest)
        ↓
[8.0s]  gather() completes ← BLOCKS UNTIL HERE
[8.0s]  NOW push all 4 to marker queue
[8.0s]  Marker finally starts processing

# Marker idle time: 8 seconds (waiting for slowest download)
```

Observable issues:
- Marker GPU 0% utilization during acquisition phase
- Memory buffer accumulating all papers before processing
- First result delayed by slowest download
- Queue sits empty until gather() completes

### Impact

- **Wasted GPU time**: Expensive GPU sits idle while waiting for network I/O
- **Increased latency**: First result delayed by slowest acquisition
- **Poor resource utilization**: Sequential instead of pipelined execution
- **Memory pressure**: All papers buffered before processing

## Root Cause

**The asyncio.gather pattern collects ALL results before returning.**

```python
# BLOCKING PATTERN
async def try_acquire_single(paper, index) -> tuple[str, Optional[str], Optional[str], bool]:
    """Returns full metadata - queue push happens AFTER gather."""
    async with semaphore:
        if source:
            return doi, None, source, is_markdown  # Just returns, doesn't push

# All tasks must complete before ANY result is processed
submit_results = await asyncio.gather(*submit_tasks, return_exceptions=True)

# ONLY NOW can we push to queue (DELAYED)
for result in submit_results:
    doi, job_id, source, is_markdown = result
    if job_id is None:
        await processing_queue.put((doi, source, paper, is_markdown))  # Too late!
```

**Why this happens:**
1. `asyncio.gather()` awaits ALL tasks before returning
2. Queue push logic was in the post-gather loop
3. Early completions wait for slowest task before being processed

## Solution

**Push to queue INSIDE the async task, before gather returns.**

### Before (Blocking)

```python
async def try_acquire_single(paper, index) -> tuple[str, Optional[str], Optional[str], bool]:
    """Full metadata returned - queue push delayed."""
    async with semaphore:
        source, is_markdown = await try_oa_download(doi, oa_url)
        if source:
            oa_acquired_count += 1
            return doi, None, source, is_markdown  # Returns without pushing

# Post-gather: process results
for result in submit_results:
    doi, job_id, source, is_markdown = result
    if job_id is None:
        # DELAYED - happens after all gather tasks complete
        await processing_queue.put((doi, source, paper, is_markdown))
```

### After (Streaming)

```python
async def try_acquire_single(paper, index) -> tuple[str, Optional[str], Optional[str]]:
    """Pushes to queue immediately on OA success - signals 'already handled'."""
    nonlocal oa_acquired_count, acquired_count

    async with semaphore:
        source, is_markdown = await try_oa_download(doi, oa_url)
        if source:
            # Push IMMEDIATELY - stream to marker while other tasks continue
            oa_acquired_count += 1
            acquired_count += 1
            acquired_paths[doi] = source
            logger.debug(f"[{acquired_count}/{total}] Acquired via OA (streaming): {doi}")

            await processing_queue.put((doi, source, paper, is_markdown))  # IMMEDIATE!

            return doi, None, None  # Signal: "OA handled, skip in post-gather"

        # Fall back to retrieve-academic
        job = await retrieve_client.submit_job(...)
        return doi, job.job_id, str(local_path)  # Needs polling

# Post-gather: only collect retrieve-academic jobs
for result in submit_results:
    doi, job_id, local_path = result
    if job_id is not None:
        # Only process items that still need polling
        valid_jobs.append((doi, job_id, local_path))
    # else: OA success, already pushed to queue
```

### Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| Queue push | After gather completes | Inside async task |
| Return value | Full metadata | Signal only |
| Marker start | After slowest download | After first download |
| GPU idle time | Full acquisition phase | ~1 second |

### Timeline (Streaming)

```
[0.0s]  Start: 4 papers to acquire
[0.5s]  OA #1 downloaded → PUSH TO QUEUE → Marker starts
[1.2s]  OA #2 downloaded → PUSH TO QUEUE → Marker processing #1, queues #2
[5.0s]  OA #3 downloaded → PUSH TO QUEUE → Marker processing #2
[8.0s]  OA #4 downloaded → PUSH TO QUEUE
[8.0s]  gather() completes
[8.0s]  Post-gather: only retrieve-academic jobs need handling
[12.0s] Marker completes (was processing throughout)

# Marker utilization: ~85% (started at 0.5s instead of 8.0s)
```

## Files Modified

- `workflows/research/academic_lit_review/paper_processor/acquisition/core.py` - Stream OA results immediately

## Prevention

### 1. Recognize the Pattern

**Blocking pattern (avoid):**
```python
results = await asyncio.gather(*tasks)  # Waits for ALL
for r in results:
    await queue.put(r)  # Delayed push
```

**Streaming pattern (prefer):**
```python
async def task():
    result = await work()
    await queue.put(result)  # Immediate push
    return signal  # Just signal, data already queued

await asyncio.gather(*tasks)  # Tasks already pushed during execution
```

### 2. Design Return Values as Signals

When tasks push directly to queues, return values become signals:

```python
# Return value indicates what POST-gather processing needs
# NOT the data itself (already in queue)

return doi, None, None      # "OA handled" - skip in post-gather
return doi, job_id, path    # "Needs polling" - collect for retrieve
```

### 3. Semaphore Placement

Semaphores should limit expensive I/O, not queue operations:

```python
async with semaphore:  # Controls concurrent DOWNLOADS
    result = await expensive_download()  # Limited concurrency here
    await queue.put(result)  # Queue push is FAST, ok to do in semaphore
```

## Related Solutions

- [Large Document Processing](../api-integration-issues/large-document-processing.md) - Memory management for pipelines
- [Workflow Reliability Retry Logic](../workflow-reliability/multi-signal-completeness-and-retry-logic.md) - Error handling in async pipelines

## Related Patterns

- [Phased Pipeline Architecture](../../patterns/data-pipeline/phased-pipeline-architecture-gpu-queue.md) - Multi-stage queue architecture
- [GPU-Accelerated Document Processing](../../patterns/data-pipeline/gpu-accelerated-document-processing.md) - Marker service integration

## References

- [Python asyncio.gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [Producer-Consumer Pattern](https://docs.python.org/3/library/asyncio-queue.html)
