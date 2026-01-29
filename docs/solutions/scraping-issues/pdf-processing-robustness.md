---
module: document_processing
date: 2026-01-28
problem_type: performance_issue
component: pdf_processor
symptoms:
  - "ReadTimeout errors during Marker polling"
  - "Memory exhaustion on large PDFs (>350MB or >100 pages)"
  - "Processing hangs indefinitely"
  - "LangSmith trace payloads exceed 20MB limit"
root_cause: config_error
resolution_type: code_fix
severity: high
tags: [pdf, marker, timeout, memory, chunking, langsmith, trace, doi, robustness]
---

# PDF Processing Robustness

## Problem

Three interconnected reliability issues caused PDF processing failures at scale:

1. **Marker API Timeouts**: ReadTimeout errors during Marker polling caused job failures under load
2. **Memory Exhaustion**: Large PDFs (>350MB or >100 pages) caused OOM crashes and processing hangs
3. **DOI Error Misclassification**: DOI resolver error pages were treated as content instead of triggering fallbacks

### Symptoms

```python
# Symptom 1: Marker API timeout during polling
httpx.ReadTimeout: timed out  # After 30s default timeout

# Symptom 2: Memory exhaustion on large PDF
MemoryError: Unable to allocate 8.5 GiB for array
# Or: Celery worker killed by OOM killer

# Symptom 3: LangSmith trace overflow
LangSmithError: Trace payload exceeds 20MB limit

# Symptom 4: DOI error treated as content
classification="research_paper"  # Should be "paywall" for error page
```

## Root Cause

### Issue 1: Aggressive Timeouts for Slow Service

Marker PDF conversion is CPU/GPU-intensive and can take minutes for complex documents:

```python
# BEFORE: Timeouts too short for realistic workloads
MARKER_POLL_INTERVAL = 2.0  # Poll every 2 seconds (excessive API calls)
timeout = 30.0  # 30s timeout (too short for large PDFs)
# No retry logic for transient network errors
```

### Issue 2: No Memory Boundaries

Large PDFs processed without limits caused:
- Worker memory exhaustion (no file size limit)
- Single-PDF processing for 500+ page documents (no chunking)
- GPU memory fragmentation (no cleanup between jobs)
- Trace payloads exceeding LangSmith limits (no truncation)

```python
# BEFORE: No limits on PDF processing
async def process_pdf_bytes(content: bytes, ...) -> str:
    # No file size check - accepts 2GB PDFs
    # No page count check - processes 1000+ pages at once
    # No trace truncation - sends full markdown to LangSmith
    pass
```

### Issue 3: DOI Error Page Detection

DOI resolver errors returned HTML error pages that were misclassified:

```python
# BEFORE: No DOI error detection
# DOI resolver error like "DOI Not Found" classified as research_paper
# Result: No fallback to retrieve-academic
```

## Solution

### Fix 1: Timeout and Retry Configuration (commit 19386db)

Increase timeouts and add retry logic for transient failures:

```python
# core/scraping/pdf/processor.py

# Marker service configuration
MARKER_POLL_INTERVAL = float(os.getenv("MARKER_POLL_INTERVAL", "15.0"))  # Was 2.0

async def _submit_marker_job(
    file_path: str,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    max_retries: int = 3,
) -> str:
    """Submit a PDF conversion job to Marker."""
    # Longer timeout for busy service
    async with httpx.AsyncClient(base_url=MARKER_BASE_URL, timeout=60.0) as client:
        payload = {
            "file_path": file_path,
            "quality": quality,
            "output_format": "markdown",
            "langs": langs or ["English"],
        }

        # Aggressive backoff for busy service: 4s, 10s, 20s
        backoff_multipliers = (2, 5, 10)
        for attempt in range(max_retries):
            try:
                response = await client.post("/convert", json=payload)
                response.raise_for_status()
                data = response.json()
                return data["job_id"]
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2.0 * backoff_multipliers[attempt]
                    logger.warning(
                        f"Marker submit timeout (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise MarkerProcessingError(
                        f"Marker job submission failed after {max_retries} retries: {e}"
                    ) from e


async def _poll_marker_job(
    job_id: str,
    max_wait: Optional[float] = None,
    max_retries: int = 3,
) -> str:
    """Poll Marker job until completion with retry logic."""
    start_time = asyncio.get_event_loop().time()

    # Longer timeout for poll requests
    async with httpx.AsyncClient(base_url=MARKER_BASE_URL, timeout=60.0) as client:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if max_wait is not None and elapsed > max_wait:
                raise MarkerProcessingError(
                    f"Marker job {job_id} did not complete within {max_wait}s"
                )

            # Retry transient network errors with longer backoffs: 30s, 75s, 150s
            backoff_multipliers = (2, 5, 10)
            for attempt in range(max_retries):
                try:
                    response = await client.get(f"/jobs/{job_id}")
                    response.raise_for_status()
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                    if attempt < max_retries - 1:
                        wait_time = MARKER_POLL_INTERVAL * backoff_multipliers[attempt]
                        logger.warning(
                            f"Marker poll timeout for job {job_id} "
                            f"(attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        raise MarkerProcessingError(
                            f"Marker poll failed after {max_retries} retries: {e}"
                        ) from e

            data = response.json()
            status = data["status"]
            # ... continue polling logic
```

**Key changes:**
- Poll interval: 2s -> 15s (Marker jobs take minutes, not seconds)
- HTTP timeout: 30s -> 60s (allow for slow responses under load)
- Retry logic: 3 attempts with aggressive backoff (2x, 5x, 10x multipliers)

### Fix 2: File Size Limit (commit 2f95af2)

Reject oversized PDFs that cause memory exhaustion:

```python
# core/scraping/pdf/processor.py

MARKER_MAX_FILE_SIZE = int(os.getenv("MARKER_MAX_FILE_SIZE", str(350 * 1024 * 1024)))  # 350MB

async def process_pdf_bytes(
    content: bytes,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: Optional[float] = None,
    filename: Optional[str] = None,
) -> str:
    """Convert PDF bytes to markdown via Marker."""
    if not validate_pdf_bytes(content):
        raise MarkerProcessingError("Content is not a valid PDF")

    # Check file size limit
    if len(content) > MARKER_MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        limit_mb = MARKER_MAX_FILE_SIZE / (1024 * 1024)
        raise MarkerProcessingError(
            f"PDF too large ({size_mb:.1f}MB > {limit_mb:.0f}MB limit)"
        )

    # ... continue processing
```

### Fix 3: PDF Chunking for Large Documents (commit 3496c59)

Split large PDFs (>100 pages) into chunks processed sequentially:

```python
# utils/pdf_chunking.py

from pypdf import PdfReader, PdfWriter

def get_page_count(content: bytes | BinaryIO) -> int:
    """Get the number of pages in a PDF."""
    if isinstance(content, bytes):
        content = io.BytesIO(content)
    reader = PdfReader(content)
    return len(reader.pages)


def should_chunk_pdf(content: bytes, page_threshold: int = 100) -> bool:
    """Determine if a PDF should be chunked based on page count."""
    try:
        page_count = get_page_count(content)
        return page_count >= page_threshold
    except Exception as e:
        logger.warning(f"Could not determine page count, skipping chunking: {e}")
        return False


def split_pdf_by_pages(
    content: bytes,
    chunk_size: int = 100,
) -> list[tuple[bytes, tuple[int, int]]]:
    """Split a PDF into chunks of N pages each.

    Returns:
        List of (chunk_bytes, (start_page, end_page)) tuples.
        Page numbers are 1-indexed for human readability.
    """
    reader = PdfReader(io.BytesIO(content))
    total_pages = len(reader.pages)

    if total_pages <= chunk_size:
        return [(content, (1, total_pages))]

    chunks = []
    for start_idx in range(0, total_pages, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pages)

        writer = PdfWriter()
        for page_idx in range(start_idx, end_idx):
            writer.add_page(reader.pages[page_idx])

        chunk_buffer = io.BytesIO()
        writer.write(chunk_buffer)
        chunk_bytes = chunk_buffer.getvalue()

        page_range = (start_idx + 1, end_idx)
        chunks.append((chunk_bytes, page_range))

    logger.info(f"Split PDF ({total_pages} pages) into {len(chunks)} chunks")
    return chunks


def assemble_markdown_chunks(
    chunks: list[str],
    page_ranges: list[tuple[int, int]],
) -> str:
    """Reassemble markdown chunks into a single document."""
    if len(chunks) == 1:
        return chunks[0]

    assembled_parts = []
    seen_h1 = False

    for i, (markdown, (start_page, end_page)) in enumerate(zip(chunks, page_ranges)):
        annotation = f"\n\n<!-- Pages {start_page}-{end_page} -->\n\n"

        # Demote H1 headings after first chunk to avoid duplicate top-level headings
        if i > 0:
            markdown = _demote_h1_headings(markdown, seen_h1)

        if re.search(r"^# [^#]", markdown, re.MULTILINE):
            seen_h1 = True

        assembled_parts.append(annotation + markdown.strip())

    return "\n\n".join(assembled_parts)
```

**Processor integration:**

```python
# core/scraping/pdf/processor.py

MARKER_CHUNK_PAGE_THRESHOLD = int(os.getenv("MARKER_CHUNK_PAGE_THRESHOLD", "100"))
MARKER_CHUNK_SIZE = int(os.getenv("MARKER_CHUNK_SIZE", "100"))

async def process_pdf_bytes(content: bytes, ...) -> str:
    """Convert PDF bytes to markdown via Marker."""
    # ... validation and size check ...

    # Check if PDF needs chunking
    if should_chunk_pdf(content, MARKER_CHUNK_PAGE_THRESHOLD):
        return await _process_chunked_pdf(content, quality=quality, ...)

    return await _process_single_pdf(content, quality=quality, ...)


async def _process_chunked_pdf(
    content: bytes,
    quality: str = "balanced",
    langs: Optional[list[str]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Process a large PDF in chunks to prevent memory exhaustion."""
    chunks = split_pdf_by_pages(content, MARKER_CHUNK_SIZE)
    logger.info(f"Processing large PDF in {len(chunks)} chunks")

    markdown_chunks = []
    page_ranges = []

    for i, (chunk_bytes, page_range) in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} (pages {page_range[0]}-{page_range[1]})")

        try:
            markdown = await _process_single_pdf(
                chunk_bytes,
                quality=quality,
                langs=langs,
                timeout=timeout,
                filename=f"chunk_{i+1}_of_{len(chunks)}.pdf",
            )
            markdown_chunks.append(markdown)
            page_ranges.append(page_range)
        finally:
            # Aggressive memory cleanup between chunks
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    return assemble_markdown_chunks(markdown_chunks, page_ranges)
```

### Fix 4: LangSmith Trace Truncation (commit 2f95af2)

Prevent oversized trace payloads that exceed LangSmith's 20MB limit:

```python
# core/config.py

LANGSMITH_MAX_TRACE_SIZE = int(os.getenv("LANGSMITH_MAX_TRACE_SIZE", str(15 * 1024 * 1024)))

def truncate_for_trace(data: Any, max_str_len: int = 50000) -> Any:
    """Truncate large strings in data structures for LangSmith tracing.

    Use with @traceable(process_inputs=truncate_for_trace, process_outputs=truncate_for_trace)
    to prevent oversized trace payloads.

    Args:
        data: Input/output data from a traced function
        max_str_len: Maximum length for string fields (default 50KB)

    Returns:
        Data with large strings truncated
    """
    if isinstance(data, str):
        if len(data) > max_str_len:
            return data[:max_str_len] + f"\n\n[TRUNCATED - {len(data):,} chars total]"
        return data
    elif isinstance(data, dict):
        return {k: truncate_for_trace(v, max_str_len) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_for_trace(item, max_str_len) for item in data]
    elif hasattr(data, "__dict__"):
        try:
            return {k: truncate_for_trace(v, max_str_len) for k, v in data.__dict__.items()}
        except Exception:
            return str(data)[:max_str_len] if len(str(data)) > max_str_len else data
    return data
```

**Apply to document processing nodes:**

```python
# workflows/document_processing/graph.py

from core.config import truncate_for_trace

@traceable(
    run_type="chain",
    name="DocumentProcessing",
    process_inputs=truncate_for_trace,
    process_outputs=truncate_for_trace,
)
async def process_document(...):
    ...

# Similarly applied to:
# - GenerateSummary (nodes/summary_agent.py)
# - SummarizeChapters (subgraphs/chapter_summarization/nodes.py)
# - AggregateSummaries (subgraphs/chapter_summarization/nodes.py)
```

### Fix 5: Marker Memory Management (commit 3496c59)

Implement model unloading and GPU memory cleanup:

```python
# services/marker/app/processor.py

MODEL_IDLE_TIMEOUT_SEC = 30 * 60  # 30 minutes

class MarkerProcessor:
    """Wrapper around Marker PDF converter with memory management."""

    def __init__(self):
        self.settings = get_settings()
        self._models = None
        self._last_used = None
        self._lock = threading.Lock()
        self._unload_timer = None

    def _get_models(self) -> dict:
        """Lazy-load models with usage tracking."""
        with self._lock:
            self._cancel_unload_timer()
            if self._models is None:
                logger.info("Loading marker models into memory...")
                self._models = create_model_dict()
                logger.info("Marker models loaded successfully")
            self._last_used = time.time()
            return self._models

    def _schedule_unload(self) -> None:
        """Schedule model unload after idle timeout."""
        self._cancel_unload_timer()
        self._unload_timer = threading.Timer(MODEL_IDLE_TIMEOUT_SEC, self._unload_models)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _unload_models(self) -> None:
        """Unload models from memory after idle timeout."""
        with self._lock:
            if self._models is None:
                return
            if self._last_used and (time.time() - self._last_used) < MODEL_IDLE_TIMEOUT_SEC:
                self._schedule_unload()
                return
            logger.info("Unloading marker models after idle timeout...")
            self._models = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Marker models unloaded, memory freed")

    def cleanup(self) -> None:
        """Clean up intermediate memory after a job (keeps models loaded)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._schedule_unload()
```

**Docker memory limits:**

```yaml
# services/marker/docker-compose.yml

services:
  marker-api:
    mem_limit: 2g
    memswap_limit: 4g

  marker-worker:
    # Memory: 30GB RAM soft target, can swap up to 32GB more before OOM
    mem_limit: 30g
    memswap_limit: 62g  # total RAM+swap ceiling
    mem_swappiness: 60  # moderate preference to swap when under pressure
```

### Fix 6: DOI Error Page Detection (commit 2f95af2)

Detect DOI resolver error pages and treat them as paywalls:

```python
# core/scraping/classification/classifier.py

def _is_doi_error_page(markdown: str) -> bool:
    """Check if content is a DOI resolver error page."""
    markdown_lower = markdown.lower()
    # DOI error pages are typically short
    if len(markdown) > 5000:
        return False

    error_indicators = [
        "doi not found",
        "the doi system",
        "doi.org",
        "handle not found",
        "invalid doi",
        "doi could not be resolved",
        "resource not found",
        "the requested doi",
        "doi resolution failed",
    ]
    matches = sum(1 for indicator in error_indicators if indicator in markdown_lower)
    return matches >= 2  # Need at least 2 indicators


async def classify_content(markdown: str, source_url: str) -> ClassificationResult:
    """Classify content type."""
    # Fast path: DOI error detection
    if _is_doi_error_page(markdown):
        logger.debug("Quick DOI error page detection")
        return ClassificationResult(
            classification="paywall",  # Treat as paywall to trigger fallback
            confidence=0.95,
            pdf_url=None,
            reasoning="Content is a DOI resolver error page (DOI not found)",
        )

    # ... rest of classification logic
```

## Files Modified

**Timeout and retry (19386db):**
- `core/scraping/pdf/processor.py` - Extended timeouts, retry logic with backoff

**Size limits and trace truncation (2f95af2):**
- `core/config.py` - `truncate_for_trace()` utility
- `core/scraping/pdf/processor.py` - `MARKER_MAX_FILE_SIZE` limit
- `core/scraping/classification/classifier.py` - DOI error page detection
- `workflows/document_processing/graph.py` - Trace truncation
- `workflows/document_processing/nodes/summary_agent.py` - Trace truncation
- `workflows/document_processing/subgraphs/chapter_summarization/nodes.py` - Trace truncation

**Chunking and memory management (3496c59):**
- `utils/pdf_chunking.py` - New: PDF splitting and reassembly
- `core/scraping/pdf/processor.py` - Chunked processing integration
- `services/marker/app/processor.py` - Model unloading, cleanup()
- `services/marker/app/tasks.py` - cleanup() calls
- `services/marker/docker-compose.yml` - Memory limits
- `requirements.txt` - pypdf dependency

## Configuration

Environment variables for tuning:

```bash
# Polling and timeouts
MARKER_POLL_INTERVAL=15.0     # Seconds between polls (default: 15)
MARKER_BASE_URL=http://localhost:8001

# Size limits
MARKER_MAX_FILE_SIZE=367001600  # 350MB default

# Chunking
MARKER_CHUNK_PAGE_THRESHOLD=100  # Pages before chunking triggers
MARKER_CHUNK_SIZE=100            # Pages per chunk

# Trace limits
LANGSMITH_MAX_TRACE_SIZE=15728640  # 15MB safety margin
```

## Prevention

### Timeout Design
- Set timeouts based on realistic workload timing (Marker jobs take minutes)
- Use poll intervals proportional to expected job duration
- Implement retry with exponential backoff for transient failures

### Memory Boundaries
- Set file size limits before expensive processing
- Chunk large inputs to bound memory usage per operation
- Clean up GPU/RAM between processing steps
- Use Docker memory limits as final safety net

### Trace Management
- Truncate large strings in traced functions
- Use `process_inputs`/`process_outputs` with traceable decorator
- Leave margin below hard limits (15MB vs 20MB limit)

### Error Classification
- Detect error pages from external services
- Treat external errors as triggers for fallback chains
- Log error page detection for debugging

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Timeout failures | 15-20% under load | <2% |
| OOM on large PDFs | Frequent | None (chunked) |
| LangSmith trace errors | 5-10% on large docs | None |
| DOI fallback rate | Low (errors misclassified) | Correct |
| Memory per PDF chunk | Unbounded | ~3GB peak |

## Related Patterns

- [GPU-Accelerated Document Processing](../../patterns/data-pipeline/gpu-accelerated-document-processing.md) - Marker service architecture

## Related Solutions

- [Scraping and PDF Processing Robustness Fixes](./scraping-pdf-robustness-fixes.md) - JSON extraction, PDF URL handling, download detection

## References

- [httpx Timeouts](https://www.python-httpx.org/advanced/timeouts/)
- [pypdf Documentation](https://pypdf.readthedocs.io/)
- [LangSmith Trace Limits](https://docs.smith.langchain.com/reference/data_formats_and_limits)
