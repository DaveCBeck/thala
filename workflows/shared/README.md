# Workflows Shared Utilities

Shared utilities and infrastructure for document processing workflows. Provides common tools for LLM integration, caching, batch processing, language support, and workflow orchestration.

## Modules

### LLM Utilities (`llm_utils/`)

Anthropic Claude integration with tiered model selection, prompt caching, and structured outputs.

- **Model Tiers**: Haiku/Sonnet/Opus selection based on task complexity
- **Prompt Caching**: 90% cost savings on repeated prompts (10% cache read vs. 100% base)
- **Structured Output**: Unified interface for schema-based LLM responses
- **Batch Processing**: 50% cost reduction via Message Batches API

Key exports:
- `get_llm(tier: ModelTier)` - Get LLM instance by tier
- `get_structured_output()` - Extract structured data from LLM
- `invoke_with_cache()` - LLM calls with prompt caching

### Token Utilities (`token_utils.py`)

Accurate token counting and budget management for Claude models.

- Fast character-based estimates with safety margins
- Accurate tiktoken-based counting for critical decisions
- Model-specific safe limits (Haiku/Sonnet/Opus)
- Request token estimation with overhead accounting

```python
from workflows.shared import (
    estimate_tokens_fast,
    count_tokens_accurate,
    check_token_budget,
    select_model_for_context,
)

# Quick estimate for budget tracking
tokens = estimate_tokens_fast(text)

# Accurate count for model selection
tokens = count_tokens_accurate(text)

# Check if within budget
check_token_budget(tokens, SONNET_SAFE_LIMIT)

# Auto-select model tier
tier = select_model_for_context(tokens)
```

### Batch Processing (`batch_processor/`)

Async batch processing via Anthropic Message Batches API. 50% cost reduction, ~1 hour completion time.

```python
from workflows.shared import BatchProcessor, ModelTier

processor = BatchProcessor()

# Queue requests
processor.add_request("task-1", "Summarize...", ModelTier.SONNET)
processor.add_request("task-2", "Analyze...", ModelTier.HAIKU)

# Execute and get results
results = await processor.execute_batch()
summary = results["task-1"]
```

### Persistent Caching (`persistent_cache.py`)

File-based caching for expensive operations with TTL support.

```python
from workflows.shared import get_cached, set_cached

# Check cache
result = get_cached("openalex", doi, ttl_days=30)
if result is None:
    result = await fetch_from_api(doi)
    set_cached("openalex", doi, result)
```

Decorator usage:
```python
from workflows.shared.persistent_cache import cached

@cached(cache_type='marker', ttl_days=7)
async def convert_pdf_to_markdown(pdf_path: str):
    # Expensive PDF processing
    return markdown
```

### Language Support (`language/`)

Multi-lingual workflow support for 30+ languages with translation and detection.

```python
from workflows.shared.language import (
    get_language_config,
    get_translated_prompt,
    translate_query,
    detect_language,
)

# Get language config
config = get_language_config("es")  # Spanish

# Translate system prompts (Opus, cached 24h)
prompt = await get_translated_prompt(
    SYSTEM_PROMPT,
    language_code="es",
    language_name="Spanish",
)

# Translate search queries (Haiku, cached 1h)
query = await translate_query(
    "machine learning",
    target_language_code="es",
)

# Detect language in content
lang_info = await detect_language(text)
```

### Text Utilities (`text_utils.py`)

Text processing utilities for document workflows.

```python
from workflows.shared import (
    count_words,
    estimate_pages,
    chunk_by_headings,
    get_first_n_pages,
)

# Basic text stats
words = count_words(markdown)
pages = estimate_pages(markdown)

# Extract sections
first_pages = get_first_n_pages(markdown, n=5)

# Smart chunking with heading preservation
chunks = chunk_by_headings(markdown, max_chunk_size=2000)
for chunk in chunks:
    print(chunk["heading"], chunk["level"], len(chunk["text"]))
```

### Chunking Utilities (`chunking_utils.py`)

Advanced document chunking with overlap for context continuity.

- Heading-based chapter detection
- Fallback chunking with ~30k word chunks
- Configurable overlap (default 500 words)
- Paragraph boundary splitting

### Workflow State Store (`workflow_state_store.py`)

Cross-workflow state sharing via persistent storage (dev mode only).

```python
from workflows.shared.workflow_state_store import (
    save_workflow_state,
    load_workflow_state,
)

# Save full state at workflow completion
save_workflow_state(
    workflow_name="academic_lit_review",
    run_id=langsmith_run_id,
    state=final_state,
)

# Load in downstream workflow
state = load_workflow_state("academic_lit_review", run_id)
paper_corpus = state.get("paper_corpus", {})
```

### Async Utilities (`async_utils.py`)

Concurrent processing with semaphore control.

```python
from workflows.shared import run_with_concurrency

# Run tasks with concurrency limit
tasks = [process_paper(p) for p in papers]
results = await run_with_concurrency(tasks, max_concurrent=5)
```

### Retry Utilities (`retry_utils.py`)

Exponential backoff retry for async operations.

```python
from workflows.shared import with_retry

result = await with_retry(
    fn=lambda: api_call(),
    max_attempts=3,
    backoff_factor=2.0,
)
```

### Node Utilities (`node_utils.py`)

Safe execution helpers for LangGraph nodes.

```python
from workflows.shared import safe_node_execution, StateUpdater

@safe_node_execution("process_papers", logger)
async def process_node(state: dict) -> dict:
    # Node logic
    return StateUpdater.success("processed", papers=results)
```

### Response Parsing (`llm_utils/response_parsing.py`)

Extract structured data from LLM responses.

```python
from workflows.shared import (
    extract_json_from_response,
    extract_response_content,
)

# Handle JSON in markdown blocks
data = extract_json_from_response(response_text)

# Extract text from various response formats
content = extract_response_content(llm_response)
```

### Workflow Wrappers (`wrappers/`)

Workflow orchestration infrastructure for dynamic registration and invocation.

```python
from workflows.shared.wrappers import (
    register_workflow,
    invoke_workflow,
    WorkflowResult,
)

# Register workflow
register_workflow(
    name="lit_review",
    workflow_fn=run_lit_review,
    supported_qualities=["quick", "standard", "comprehensive"],
)

# Invoke registered workflow
result = await invoke_workflow(
    workflow_name="lit_review",
    quality_tier="standard",
    **params,
)
```

## Usage

Import utilities directly from the shared module:

```python
from workflows.shared import (
    # LLM
    ModelTier,
    get_llm,
    get_structured_output,
    # Tokens
    estimate_tokens_fast,
    check_token_budget,
    # Caching
    get_cached,
    set_cached,
    # Text
    count_words,
    chunk_by_headings,
    # Async
    run_with_concurrency,
    with_retry,
)
```

## API Reference

### Constants

**Token Limits:**
- `HAIKU_SAFE_LIMIT` - 150,000 tokens (75% of max)
- `SONNET_SAFE_LIMIT` - 150,000 tokens
- `SONNET_1M_SAFE_LIMIT` - 800,000 tokens
- `CHARS_PER_TOKEN` - 4 (conservative estimate)
- `DEFAULT_RESPONSE_BUFFER` - 4,096 tokens

**Cache:**
- `CACHE_DIR` - `~/.thala/.cache` (configurable via `THALA_CACHE_DIR`)
- `CACHE_DISABLED` - Set `THALA_CACHE_DISABLED=1` to disable all caching

### Key Functions

**LLM Integration:**
- `get_llm(tier: ModelTier)` - Get LLM by tier
- `get_structured_output()` - Unified structured output interface
- `invoke_with_cache()` - LLM with prompt caching

**Token Management:**
- `estimate_tokens_fast(text: str)` - Quick estimate
- `count_tokens_accurate(text: str)` - Accurate tiktoken count
- `estimate_request_tokens()` - Full request estimation
- `check_token_budget(tokens, limit)` - Budget validation
- `select_model_for_context(tokens)` - Auto tier selection

**Caching:**
- `get_cached(cache_type, key, ttl_days)` - Retrieve cached value
- `set_cached(cache_type, key, value)` - Save to cache
- `clear_cache(cache_type)` - Clear cache by type

**Text Processing:**
- `count_words(text)` - Word count
- `estimate_pages(text)` - Page count estimate
- `chunk_by_headings(markdown)` - Heading-aware chunking
- `get_first_n_pages(markdown, n)` - Extract first N pages
- `get_last_n_pages(markdown, n)` - Extract last N pages

**Async/Retry:**
- `run_with_concurrency(tasks, max_concurrent)` - Concurrent execution
- `with_retry(fn, max_attempts)` - Retry with backoff

**State Management:**
- `safe_node_execution(node_name, logger)` - Decorator for safe nodes
- `StateUpdater.success(status, **kwargs)` - Success state
- `StateUpdater.error(node_name, error)` - Error state
