# Logging Configuration

## Overview

Thala uses a **module-based logging system** with **run-based rotation**. This provides:

- **Module dispatch**: Logs are automatically routed to workflow-specific files (e.g., `stores.log`, `lit-review.log`)
- **Run-based rotation**: Logs rotate on first write per run, keeping `current.log` and `previous.log`
- **Third-party isolation**: Noisy library logs go to `run-3p.log`
- **Async-safe**: Uses ContextVars for isolation between concurrent async tasks
- **Thread-safe**: Explicit locks prevent race conditions in concurrent scenarios

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `THALA_LOG_LEVEL_CONSOLE` | Console output level | `WARNING` |
| `THALA_LOG_LEVEL_FILE` | File output level | `INFO` |
| `THALA_LOG_DIR` | Log file directory | `./logs/` |

## Log Files

Logs are stored in `./logs/` with module-based separation:

```
logs/
├── stores.log              # core.stores.* modules
├── stores.previous.log     # Previous run's stores logs
├── task-queue.log          # core.task_queue.* modules
├── lit-review.log          # workflows.research.academic_lit_review.*
├── web-research.log        # workflows.research.web_research.*
├── supervision.log         # workflows.enhance.supervision.*
├── scraping.log            # core.scraping.* modules
├── run-3p.log              # All third-party library logs
├── run-3p.previous.log
└── misc.log                # Unmapped modules (fallback)
```

### Module-to-Log Mapping

Modules are routed to log files based on longest-prefix match:

| Module Prefix | Log File |
|---------------|----------|
| `workflows.research.academic_lit_review` | `lit-review.log` |
| `workflows.research.web_research` | `web-research.log` |
| `workflows.enhance.supervision` | `supervision.log` |
| `workflows.enhance.editing` | `editing.log` |
| `workflows.document_processing` | `doc-processing.log` |
| `core.stores` | `stores.log` |
| `core.task_queue` | `task-queue.log` |
| `core.scraping` | `scraping.log` |
| `core.images` | `images.log` |
| `testing` | `testing.log` |
| *(unmapped)* | `misc.log` |

### Rotation Strategy

- **Two-file rotation**: Each log keeps `{name}.log` (current) and `{name}.previous.log`
- **Run-triggered**: Rotation happens on first write to each log file per run
- **Atomic**: Previous is deleted, current renamed to previous, new file opened

## Third-Party Loggers

These libraries log to `run-3p.log` (separate from application logs):

- `httpx`, `httpcore`, `urllib3`
- `asyncio`
- `elasticsearch`
- `chromadb`
- `openai`, `anthropic`
- `hpack`, `charset_normalizer`, `filelock`, `fsspec`, `huggingface_hub`

**Note**: `langchain` and `langgraph` logs are kept in main log files for workflow visibility.

## Usage

### Entry Points (services, CLI tools)

Call `configure_logging()` once at startup:

```python
from core.config import configure_logging, configure_langsmith

configure_logging()  # Sets up module-based dispatch
configure_langsmith()
```

### Run Lifecycle (task queue, tests)

Wrap work units with `start_run()` / `end_run()` to trigger rotation:

```python
from core.logging import start_run, end_run

start_run("task-abc123")  # Triggers rotation on first log per module
try:
    # ... do work ...
finally:
    end_run()
```

This is **already handled** by:
- Task queue runner (wraps each task execution)
- Test fixtures in `conftest.py` (wraps each test module)

### All Other Modules

Use the standard pattern (no changes needed):

```python
import logging
logger = logging.getLogger(__name__)

# Logs automatically route to the correct file based on __name__
logger.debug("Cache hit for DOI %s", doi)
logger.info("Starting search for query: %s", query)
logger.warning("Primary search failed, using fallback")
logger.error("Search failed: %s", error)
```

## Format

**Console** (compact):
```
WARNING - core.scraping.pdf - PDF extraction failed for https://example.com
```

**File** (detailed with timestamp):
```
2026-01-13 14:30:52,123 - core.scraping.pdf - WARNING - PDF extraction failed for https://example.com
```

---

## Logging Conventions

### Log Levels

| Level | Use For | Examples |
|-------|---------|----------|
| **DEBUG** | Internal flow, cache operations, verbose detail | `"Cache hit for DOI {doi}"`, `"Parsed {n} results"` |
| **INFO** | Significant operations, workflow milestones | `"Starting search for {query}"`, `"Workflow complete"` |
| **WARNING** | Recoverable issues, fallbacks triggered | `"Primary search failed, using fallback"`, `"Rate limited"` |
| **ERROR** | Failures that affect results | `"Search failed: {e}"`, `"Unable to connect"` |

### Guidelines

**DO:**
- Include relevant context: `logger.info(f"Processing {len(items)} items for '{query}'")`
- Include exceptions in errors: `logger.error(f"Failed to fetch {url}: {e}")`
- Use `logger.exception()` when you need the full traceback

**DON'T:**
- Log sensitive data (API keys, credentials, tokens in URLs)
- Log redundant info (same data in caller and callee)
- Use INFO for routine success paths (use DEBUG instead)
- Use print() for debugging (use logger.debug())
- Create loggers with dynamic names (e.g., `f"module.{task_id}"`) - use `__name__`

### What to Log at Each Level

**DEBUG** (verbose, for troubleshooting):
- Cache hits/misses
- Local vs cloud fallback decisions
- Intermediate processing steps
- API response summaries

**INFO** (operational visibility):
- Service/workflow start and completion
- Major phase transitions
- Configuration being used
- Summary counts (e.g., "Found 15 papers")

**WARNING** (needs attention but not broken):
- Fallback paths triggered
- Rate limiting encountered
- Missing optional data
- Deprecated usage detected

**ERROR** (something failed):
- API/service failures
- Required data missing
- Unrecoverable errors (with exception details)

---

## Architecture Notes

### Async Safety

The logging system uses `ContextVar` for run tracking, ensuring:
- Concurrent async tasks have isolated rotation state
- No cross-task log pollution
- Safe to use with `asyncio.gather()` and task queues

### Thread Safety

A `threading.Lock` protects the rotation check-and-add operation:
- Prevents double rotation when threads share ContextVar context
- Safe to use with `concurrent.futures.ThreadPoolExecutor`

### Blocking I/O

The handlers perform synchronous file I/O (write, flush, unlink, rename). While this technically blocks the event loop when called from async code, the impact is typically 100-1000 microseconds per operation, which is acceptable for most workloads. For high-throughput async applications with strict latency requirements, consider using `logging.handlers.QueueHandler`.

### Security

The `_compute_log_name()` function falls back to `"misc"` for unmapped modules, preventing arbitrary file creation via malicious logger names. Never use module names directly in file paths without this sanitization layer.
