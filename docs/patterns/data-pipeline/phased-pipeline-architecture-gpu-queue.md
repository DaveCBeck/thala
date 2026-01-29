---
name: phased-pipeline-architecture-gpu-queue
title: Phased Pipeline Architecture for GPU Queue
date: 2026-01-02
category: data-pipeline
applicability:
  - "Processing pipelines with external API rate limits"
  - "GPU queues that benefit from batch submission"
  - "Mixed I/O-bound and compute-bound operations"
  - "Optimizing worker utilization across rate-limited operations"
components: [paper_processor, marker_client]
complexity: medium
verified_in_production: true
tags: [pipeline, gpu, celery, rate-limiting, batch-processing, parallelization]
---

# Phased Pipeline Architecture for GPU Queue

## Intent

Separate a processing pipeline into distinct phases (cache check, acquisition, processing) to maximize GPU worker utilization by keeping the task queue full while respecting external API rate limits.

## Problem

Sequential per-item processing creates inefficiencies:

```
# Problem: Sequential pipeline
for paper in papers:
    cached = check_cache(paper)      # 10ms
    if not cached:
        pdf = acquire(paper)          # 2s (rate-limited)
        result = process(pdf)         # 60s (GPU)
    # GPU workers idle during check_cache and acquire
```

Issues:
- **Worker idle time**: GPU workers wait while checking cache and downloading
- **Queue starvation**: Only one job at a time in the Celery queue
- **Rate limit interference**: Processing time affects acquisition rate
- **No parallelism**: Different operations can't overlap

## Solution

Separate into three phases with phase-appropriate parallelism:

1. **Phase 1 - Cache Check**: Parallel, fast (all papers at once)
2. **Phase 2 - Acquisition**: Rate-limited, sequential-ish (external API)
3. **Phase 3 - Processing**: Parallel, high concurrency (fills GPU queue)

```
Phase 1 (parallel):      ████████████████████████  Cache check all papers
Phase 2 (rate-limited):  ══►  ══►  ══►  ══►  ══►  Acquire PDFs
Phase 3 (parallel):      ████  ████  ████  ████   Process through Marker
                              └──GPU queue stays full──┘
```

## Implementation

### Phase 1: Parallel Cache Check

Check all papers against cache simultaneously:

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py

async def check_cache_for_paper(paper: PaperMetadata) -> tuple[str, Optional[dict]]:
    """Check if paper exists in cache.

    Returns:
        Tuple of (doi, cached_result or None)
    """
    doi = paper.get("doi")
    existing = await check_document_exists_by_doi(doi)
    if existing:
        return doi, {
            "doi": doi,
            "success": True,
            "es_record_id": existing["es_record_id"],
            "zotero_key": existing["zotero_key"],
            "short_summary": existing["short_summary"],
            "errors": [],
        }
    return doi, None


async def run_paper_pipeline(papers: list[PaperMetadata], max_concurrent: int = 2):
    """Run phased pipeline."""

    # Phase 1: Check cache for ALL papers (parallel, fast)
    logger.info(f"Phase 1: Checking cache for {len(papers)} papers...")
    cache_tasks = [check_cache_for_paper(p) for p in papers]
    cache_results = await asyncio.gather(*cache_tasks)

    cached_results = {}
    papers_to_acquire = []
    for doi, cached in cache_results:
        if cached:
            cached_results[doi] = cached
            logger.info(f"Cache hit: {doi}")
        else:
            papers_to_acquire.append(papers_by_doi[doi])

    logger.info(
        f"Phase 1 complete: {len(cached_results)} cached, "
        f"{len(papers_to_acquire)} need acquisition"
    )
```

### Phase 2: Rate-Limited Acquisition

Download PDFs with external API rate limiting:

```python
    # Phase 2: Acquire all non-cached PDFs (rate-limited)
    acquired_paths = {}  # doi -> local_path
    acquisition_failed = []

    if papers_to_acquire:
        logger.info(f"Phase 2: Acquiring {len(papers_to_acquire)} PDFs...")
        semaphore = asyncio.Semaphore(max_concurrent)  # Low concurrency
        ACQUISITION_DELAY = 2.0  # Delay between requests

        async def acquire_with_limit(paper: PaperMetadata, index: int) -> tuple[str, Optional[str]]:
            async with semaphore:
                if index > 0:
                    await asyncio.sleep(ACQUISITION_DELAY)  # Rate limiting
                doi = paper.get("doi")
                path = await acquire_full_text(paper, client, output_dir)
                return doi, path

        acquire_tasks = [
            acquire_with_limit(p, i) for i, p in enumerate(papers_to_acquire)
        ]
        acquire_results = await asyncio.gather(*acquire_tasks, return_exceptions=True)

        for result in acquire_results:
            if isinstance(result, Exception):
                logger.error(f"Acquisition task failed: {result}")
                continue
            doi, path = result
            if path:
                acquired_paths[doi] = path
            else:
                acquisition_failed.append(doi)

        logger.info(
            f"Phase 2 complete: {len(acquired_paths)} acquired, "
            f"{len(acquisition_failed)} failed"
        )
```

### Phase 3: Parallel Processing

Process all acquired PDFs with high concurrency to fill GPU queue:

```python
    # Phase 3: Process all acquired PDFs (parallel, fills marker queue)
    MAX_PROCESSING_CONCURRENT = 4  # Higher than acquisition
    processing_results = dict(cached_results)  # Start with cached
    processing_failed = []

    if acquired_paths:
        logger.info(f"Phase 3: Processing {len(acquired_paths)} documents...")
        process_semaphore = asyncio.Semaphore(MAX_PROCESSING_CONCURRENT)
        completed_count = 0
        total_to_process = len(acquired_paths)

        async def process_with_limit(doi: str, path: str) -> tuple[str, dict]:
            nonlocal completed_count
            async with process_semaphore:
                paper = papers_by_doi[doi]
                result = await process_single_document(doi, path, paper)
                completed_count += 1
                title = paper.get("title", "Unknown")[:50]
                if result.get("success"):
                    logger.info(f"[{completed_count}/{total_to_process}] Completed: {title}")
                else:
                    logger.warning(f"[{completed_count}/{total_to_process}] Failed: {title}")
                return doi, result

        process_tasks = [
            process_with_limit(doi, path) for doi, path in acquired_paths.items()
        ]
        process_results = await asyncio.gather(*process_tasks, return_exceptions=True)

        for result in process_results:
            if isinstance(result, Exception):
                logger.error(f"Processing task failed: {result}")
                continue
            doi, proc_result = result
            if proc_result.get("success"):
                processing_results[doi] = proc_result
            else:
                processing_failed.append(doi)

        logger.info(
            f"Phase 3 complete: {len(processing_results) - len(cached_results)} processed, "
            f"{len(processing_failed)} failed"
        )

    return acquired_paths, processing_results, acquisition_failed, processing_failed
```

### Marker Client Batch Methods

Add batch submission and polling to the Marker client:

```python
# workflows/shared/marker_client.py

class MarkerClient:
    async def submit_jobs(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[str]:
        """Submit multiple document conversion jobs.

        Args:
            jobs: List of job configs with file_path, quality, langs

        Returns:
            List of job IDs for polling status
        """
        job_ids = []
        for job in jobs:
            job_id = await self.submit_job(
                file_path=job["file_path"],
                quality=job.get("quality", "balanced"),
                langs=job.get("langs"),
            )
            job_ids.append(job_id)
        return job_ids

    async def poll_job_status(self, job_id: str) -> dict[str, Any]:
        """Get current status of a job without blocking."""
        client = await self._get_client()
        response = await client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    async def poll_multiple_until_complete(
        self,
        job_ids: list[str],
        max_wait: float | None = None,
    ):
        """Poll multiple jobs, yielding results as they complete.

        Async generator that yields (job_id, result) tuples as jobs finish.
        This allows processing results as they stream in.
        """
        pending = set(job_ids)
        start_time = asyncio.get_event_loop().time()

        while pending:
            if max_wait and (asyncio.get_event_loop().time() - start_time) > max_wait:
                raise TimeoutError(f"Jobs not complete after {max_wait}s: {pending}")

            for job_id in list(pending):
                status = await self.poll_job_status(job_id)
                if status.get("status") == "completed":
                    pending.remove(job_id)
                    yield job_id, status.get("result")
                elif status.get("status") == "failed":
                    pending.remove(job_id)
                    yield job_id, {"error": status.get("error")}

            if pending:
                await asyncio.sleep(5)  # Poll interval
```

### Worker Scaling Configuration

Scale Marker workers to share GPU:

```yaml
# services/marker/docker-compose.yml

services:
  marker-worker:
    environment:
      # Reduced batch sizes to share GPU between workers
      - INFERENCE_RAM=12          # Was 24
      - RECOGNITION_BATCH_SIZE=128  # Was 192
      - DETECTOR_BATCH_SIZE=12      # Was 18
      - LAYOUT_BATCH_SIZE=16        # Was 24
      - TABLE_REC_BATCH_SIZE=24     # Was 36
    deploy:
      replicas: 2  # Two workers sharing GPU
```

## Usage

```python
from workflows.research.subgraphs.academic_lit_review.paper_processor import run_paper_pipeline

# Run phased pipeline
acquired, results, acq_failed, proc_failed = await run_paper_pipeline(
    papers=papers_to_process,
    max_concurrent=2,  # Acquisition concurrency (rate-limited)
)

# Processing concurrency is internal (MAX_PROCESSING_CONCURRENT = 4)
# This keeps 4 jobs in the Marker queue at all times
```

## Guidelines

### Phase Concurrency Settings

| Phase | Concurrency | Rationale |
|-------|-------------|-----------|
| Cache check | Unlimited | Fast, local ES queries |
| Acquisition | 2 | Rate-limited external APIs |
| Processing | 4 | Fills GPU queue, 2 workers |

### Scaling Considerations

- **GPU memory**: Reduce batch sizes when scaling workers
- **Queue depth**: Processing concurrency = workers × 2 (keeps queue full)
- **Rate limits**: Acquisition concurrency independent of processing

### Logging Best Practices

Log phase transitions and progress:
```python
logger.info(f"Phase 1 complete: {len(cached)} cached, {len(to_acquire)} need acquisition")
logger.info(f"Phase 2 complete: {len(acquired)} acquired, {len(failed)} failed")
logger.info(f"[{count}/{total}] Completed: {title[:50]}")
```

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/paper_processor/acquisition.py`
- `services/marker/docker-compose.yml` - 2 worker replicas
- `workflows/shared/marker_client.py` - Batch submission methods

## Consequences

### Benefits
- **Zero idle time**: GPU workers always have work
- **Independent rate limiting**: Acquisition doesn't block processing
- **Early results**: Cached items available immediately
- **Better throughput**: Parallel processing of acquired PDFs

### Trade-offs
- **Memory usage**: All PDFs downloaded before processing starts
- **Complexity**: Three separate phase implementations
- **Disk space**: Temporary PDF storage during Phase 2

## Related Patterns

- [Paper Acquisition Robustness](../../solutions/api-integration-issues/paper-acquisition-robustness.md) - ES cache check
- [GPU-Accelerated Document Processing Service](./gpu-document-processing-service.md) - Marker service

## References

- [Celery Best Practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html#best-practices)
- [asyncio.gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
