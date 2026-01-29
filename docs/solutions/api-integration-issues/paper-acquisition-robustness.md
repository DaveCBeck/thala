---
module: paper_processor, retrieve_academic
date: 2025-12-31
problem_type: api_integration_issue
component: paper_processor, retrieve_academic, marker_processor, zotero_stub
symptoms:
  - "Single-source download failures causing papers to be missed"
  - "Duplicate processing of already-cached documents"
  - "Marker timeout on large PDFs causing processing failures"
  - "No automatic recovery when primary download source fails"
root_cause: architecture_issue
resolution_type: code_fix
severity: high
tags: [paper-acquisition, caching, elasticsearch, unpaywall, retrieve-academic, checkpoint, pdf-processing]
---

# Paper Acquisition Robustness

## Problem

The academic literature review workflow experienced multiple reliability issues during paper acquisition and processing:

1. **No local cache check**: Papers were re-downloaded even when already processed and stored in Elasticsearch
2. **Single-source dependency**: Relied solely on retrieve-academic without OA sources, failing when papers weren't available
3. **No retry mechanism**: Download failures were permanent with no alternative source fallback
4. **Marker timeout**: Large PDFs timed out after 20 minutes, failing processing
5. **No workflow recovery**: Failed runs required full restart from discovery phase

### Symptoms

```
# Duplicate downloads for cached papers
INFO: Processing: Neural Network Architecture... (already exists in ES L0)
WARNING: Downloaded 45MB PDF that was already processed

# Single source failures
ERROR: httpx.HTTPStatusError: 400 Client Error for url: /download/1234567
WARNING: Paper acquisition failed for 10.1038/nature12345 (no fallback)

# Marker timeout on large PDFs
ERROR: TimeoutError: Marker processing exceeded 20 minutes
ERROR: Processing failed for 500-page textbook

# Full restart required after failures
INFO: Resuming requires re-running 6-hour diffusion phase
```

## Root Cause

### Issue 1: No Elasticsearch Cache Check

Papers were acquired without checking if they already existed in the local store:

```python
# BEFORE: Always download regardless of cache state
async def acquire_and_process_single_paper(paper, client, output_dir, ...):
    # Immediately attempts download
    local_path = await acquire_full_text(paper, client, output_dir)
    # Downloads 45MB PDF that was already in ES L0
```

### Issue 2: Single Source Without Fallback

The retrieve-academic service only tried one source per DOI:

```python
# BEFORE: Single attempt, no alternatives
async def download_paper(doi: str) -> Path | None:
    source_id = search_results[0]["source_id"]  # First result only
    return await download_from_source(source_id)  # Fails permanently on 400
```

### Issue 3: Fixed Marker Timeout

Marker had a 20-minute timeout that was too short for large academic books:

```python
# BEFORE: Hard-coded timeout
MARKER_TIMEOUT = 1200  # 20 minutes
# 500-page textbook fails at 19 minutes of processing
```

### Issue 4: No Checkpoint Support

Failed runs required full workflow restart:

```python
# BEFORE: Monolithic execution
result = await academic_lit_review(topic, questions, quality)
# 6-hour diffusion phase runs again after processing failure
```

## Solution

### Step 1: Add Elasticsearch L0 Cache Check

Check if paper already exists in ES before attempting download:

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor.py

async def check_document_exists_by_doi(doi: str) -> Optional[dict[str, Any]]:
    """Check if document already exists in ES L0 by DOI.

    Returns dict with es_record_id, zotero_key, short_summary, content if found.
    """
    store_manager = get_store_manager()

    try:
        results = await store_manager.es_stores.store.search(
            query={
                "bool": {
                    "must": [
                        {"term": {"metadata.doi": doi}},
                        {"term": {"metadata.processing_status": "completed"}}
                    ]
                }
            },
            size=1,
            compression_level=0,  # Search L0 only
        )

        if results:
            record = results[0]
            return {
                "es_record_id": str(record.id),
                "zotero_key": record.zotero_key,
                "content": record.content,
                "short_summary": record.metadata.get("short_summary", ""),
            }
    except Exception as e:
        logger.debug(f"ES lookup for DOI {doi} failed: {e}")

    return None
```

Update the acquisition function to check cache first:

```python
async def acquire_and_process_single_paper(
    paper: PaperMetadata,
    client: RetrieveAcademicClient,
    output_dir: Path,
    paper_index: int,
    total_papers: int,
) -> dict[str, Any]:
    doi = paper.get("doi")
    result = {
        "doi": doi,
        "acquired": False,
        "processing_success": False,
        "from_cache": False,  # NEW: Track cache hits
    }

    # Step 0: Check if document already exists in local store
    existing = await check_document_exists_by_doi(doi)
    if existing:
        logger.info(f"[{paper_index}/{total_papers}] Cache hit for {doi}, skipping download")
        result["acquired"] = True
        result["processing_success"] = True
        result["from_cache"] = True
        result["processing_result"] = {
            "doi": doi,
            "success": True,
            "es_record_id": existing["es_record_id"],
            "zotero_key": existing["zotero_key"],
            "short_summary": existing["short_summary"],
            "errors": [],
        }
        return result

    # Step 1: Acquire full text (only if not cached)
    local_path = await acquire_full_text(paper, client, output_dir)
    # ... rest of processing
```

Store DOI in Zotero stub metadata for cache lookups:

```python
# workflows/document_processing/nodes/zotero_stub.py

async def create_zotero_stub(state: dict) -> dict:
    # ... existing code ...
    es_metadata = {
        "title": title,
        "processing_status": "pending",
        "source": input_data["source"],
        "doi": input_data.get("extra_metadata", {}).get("DOI"),  # NEW: Store DOI
    }
```

### Step 2: Add Multi-Source Acquisition with Retry

Update retrieve-academic to try alternative sources on failure:

```python
# services/retrieve-academic/app/retriever.py

MAX_SOURCE_ATTEMPTS = 5

async def download_paper(doi: str) -> Path | None:
    """Download paper with multi-source fallback.

    Tries up to 5 alternative source IDs from search results
    when primary source fails.
    """
    search_results = await search_sources(doi)
    if not search_results:
        return None

    # Try multiple sources on failure
    for i, result in enumerate(search_results[:MAX_SOURCE_ATTEMPTS]):
        source_id = result["source_id"]
        try:
            path = await download_from_source(source_id)
            if path and path.exists():
                logger.info(f"Downloaded from source {i+1}/{len(search_results)}")
                return path
        except HTTPStatusError as e:
            if e.response.status_code == 400:
                logger.warning(f"Source {i+1} failed (400), trying alternative...")
                continue
            raise

    logger.error(f"All {MAX_SOURCE_ATTEMPTS} sources failed for {doi}")
    return None
```

Add Unpaywall as primary OA source:

```python
# Unpaywall provides legal open access URLs
async def get_unpaywall_url(doi: str) -> str | None:
    """Check Unpaywall for legal OA PDF URL."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    response = await client.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("is_oa") and data.get("best_oa_location"):
            return data["best_oa_location"].get("url_for_pdf")
    return None
```

### Step 3: Remove Marker Timeout

Allow Marker to run without timeout for large PDFs:

```python
# workflows/shared/marker_client.py

# BEFORE: Hard timeout
# timeout = httpx.Timeout(1200.0)  # 20 minutes

# AFTER: No timeout (let GPU queue manage)
timeout = httpx.Timeout(None)  # No timeout

# Marker GPU queue handles prioritization and resource management
# Large PDFs may take hours but will complete
```

### Step 4: Add Checkpoint Support for Workflow Recovery

Add checkpoint save/load functions for resuming from expensive phases:

```python
# testing/test_academic_lit_review.py

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def save_checkpoint(state: dict, name: str) -> Path:
    """Save workflow state to a checkpoint file for later resumption."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info(f"Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(name: str) -> dict | None:
    """Load workflow state from a checkpoint file."""
    checkpoint_file = CHECKPOINT_DIR / f"{name}.json"
    if not checkpoint_file.exists():
        return None
    with open(checkpoint_file, "r") as f:
        return json.load(f)
```

Add phase-by-phase execution with automatic checkpointing:

```python
async def run_with_checkpoints(
    topic: str,
    research_questions: list[str],
    quality: str = "quick",
    date_range: tuple[int, int] | None = None,
    checkpoint_prefix: str = "latest",
) -> dict:
    """Run full workflow with automatic checkpoint saves after expensive phases."""
    from workflows.research.subgraphs.academic_lit_review.graph import (
        discovery_phase_node,
        diffusion_phase_node,
        processing_phase_node,
        clustering_phase_node,
        synthesis_phase_node,
    )

    state = build_initial_state(topic, research_questions, quality, date_range)

    # Phase 1: Discovery
    state.update(await discovery_phase_node(state))

    # Phase 2: Diffusion (expensive - API calls, relevance scoring)
    state.update(await diffusion_phase_node(state))
    save_checkpoint(state, f"{checkpoint_prefix}_after_diffusion")

    # Phase 3: Processing (expensive - PDF download, Marker, LLM summaries)
    state.update(await processing_phase_node(state))
    save_checkpoint(state, f"{checkpoint_prefix}_after_processing")

    # Phase 4: Clustering
    state.update(await clustering_phase_node(state))

    # Phase 5: Synthesis
    state.update(await synthesis_phase_node(state))

    return state


async def run_from_diffusion_checkpoint(checkpoint_prefix: str) -> dict:
    """Resume workflow from after-diffusion checkpoint.

    Runs: processing -> clustering -> synthesis
    Skips: discovery, diffusion (saves hours on retry)
    """
    state = load_checkpoint(f"{checkpoint_prefix}_after_diffusion")
    if not state:
        raise ValueError(f"Checkpoint not found")

    logger.info(f"Paper corpus size: {len(state.get('paper_corpus', {}))}")

    state.update(await processing_phase_node(state))
    save_checkpoint(state, f"{checkpoint_prefix}_after_processing")

    state.update(await clustering_phase_node(state))
    state.update(await synthesis_phase_node(state))

    return state
```

## Prevention

### Cache Check Guidelines

1. **Always store DOI in metadata** when creating document stubs
2. **Check ES L0 before downloading** any paper by DOI
3. **Track cache hits** in pipeline statistics for monitoring
4. **Query with `compression_level=0`** to search only the L0 store

### Multi-Source Acquisition Guidelines

1. **Try multiple sources** (at least 5) before failing permanently
2. **Use Unpaywall first** for legal OA access
3. **Log which source succeeded** for debugging and optimization
4. **Handle 400 errors gracefully** - they often mean "try another source"

### Timeout Guidelines

1. **Avoid hard timeouts** for processing large documents
2. **Let the GPU queue manage** resource allocation
3. **Use task queues with deadlines** instead of HTTP timeouts
4. **Monitor processing time** but don't fail on long-running tasks

### Checkpoint Guidelines

1. **Save checkpoints after expensive phases** (diffusion, processing)
2. **Use JSON serialization** with `default=str` for datetime handling
3. **Include all state fields** needed to resume
4. **Provide resume functions** for each checkpoint point

## Files Modified

- `workflows/document_processing/nodes/zotero_stub.py`: Store DOI in ES metadata
- `workflows/research/subgraphs/academic_lit_review/paper_processor.py`: Cache check, cache hit tracking
- `services/retrieve-academic/app/retriever.py`: Multi-source retry logic
- `workflows/shared/marker_client.py`: Remove timeout
- `testing/test_academic_lit_review.py`: Checkpoint save/load, phase-by-phase execution

## Related Patterns

- [Citation Network Academic Review Workflow](../../patterns/langgraph/citation-network-academic-review-workflow.md) - Overall workflow architecture
- [Concurrent Scraping with TTL Cache](../../patterns/async-python/concurrent-scraping-with-ttl-cache.md) - Caching patterns for external APIs

## Related Solutions

- [Academic Literature Review Reliability Fixes](./academic-lit-review-reliability-fixes.md) - Rate limiting and unified pipeline

## References

- [Unpaywall API Documentation](https://unpaywall.org/products/api)
- [Elasticsearch Term Query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-term-query.html)
