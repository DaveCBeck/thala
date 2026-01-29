---
module: document_processing, chapter_summarization, paper_processor
date: 2025-12-31
problem_type: resource_exhaustion
component: chapter_detector, chapter_summarization, marker, diffusion_engine
symptoms:
  - "LLM chapter detection fails on documents with unusual heading structures"
  - "Token limit errors when processing very long chapters (>600k chars)"
  - "No visibility into memory usage during Marker GPU processing"
  - "40k character truncation losing most content in long documents"
  - "JSON parsing failures from LLM chapter analysis"
root_cause: architecture_issue
resolution_type: code_fix
severity: high
tags: [large-documents, memory-monitoring, chunking, token-limits, structured-output, fallback-handling]
---

# Large Document Processing Solutions

## Problem

Processing large documents (books, long papers) caused multiple failure modes:

1. **LLM chapter detection failures**: JSON parsing errors when LLM returned malformed chapter analysis
2. **Token limit errors**: Very long chapters (>600k chars) exceeded context window limits
3. **Content truncation**: 40k character limit lost most content from long documents
4. **No memory visibility**: GPU/RAM usage during Marker processing was opaque
5. **Relevance filtering**: No mechanism to filter corpus to top papers by quality

### Symptoms

```
# JSON parsing failure in chapter detection
ERROR: Failed to parse chapter analysis JSON: Expecting property name enclosed in double quotes
ERROR: Chapter detection failed, skipping 10:1 summary

# Token limit errors on large chapters
ERROR: Request too large: 180000 tokens exceeds maximum of 200000
ERROR: Chapter summarization failed for "Complete Guide to Neural Networks"

# Content truncation
WARNING: Truncating chapter content from 250000 to 40000 chars
# Summary only covers first 16% of chapter

# No memory visibility
INFO: Processing 500-page book...
# No indication of memory pressure until OOM kill
```

## Root Cause

### Issue 1: Fragile JSON Extraction

Chapter detection used prompt-based JSON that failed on malformed LLM output:

```python
# BEFORE: Prompt-based JSON extraction
analysis = await extract_json(
    text=heading_list,
    prompt=prompt,
    schema_hint=schema_hint,  # Just a hint, not enforced
)
# LLM returns malformed JSON, entire chapter detection fails
```

### Issue 2: No Chunking for Large Chapters

Chapter summarization assumed all chapters fit within token limits:

```python
# BEFORE: No size checking
chapter_content = markdown[chapter["start_position"]:chapter["end_position"]]
# 800k character chapter exceeds 200k token limit
response = await llm.ainvoke(user_prompt)  # Fails with token limit error
```

### Issue 3: Aggressive Content Truncation

Extraction truncated content to fit context window:

```python
# BEFORE: Hard truncation
MAX_CHARS = 40000
content = chapter_content[:MAX_CHARS]  # Loses 84% of 250k char chapter
```

### Issue 4: No Memory Monitoring

Marker Celery tasks had no visibility into resource usage:

```python
# BEFORE: No monitoring
def convert_document(file_path: str) -> dict:
    processor = get_processor()
    return processor.process(file_path)  # Unknown memory usage
```

## Solution

### Step 1: Add Structured Output with Tool Use

Replace prompt-based JSON extraction with guaranteed-valid structured output:

```python
# workflows/shared/llm_utils.py

async def extract_structured(
    text: str,
    prompt: str,
    schema: dict,
    tier: ModelTier = ModelTier.SONNET,
) -> dict:
    """Extract structured data using Anthropic tool use.

    Uses tool_choice="required" to guarantee valid JSON matching schema.
    """
    llm = get_llm(tier=tier)

    # Define extraction tool with schema
    tools = [{
        "name": "extract_data",
        "description": "Extract structured data from text",
        "input_schema": schema,
    }]

    messages = [
        {"role": "user", "content": f"{prompt}\n\n{text}"}
    ]

    response = await llm.ainvoke(
        messages,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_data"},
    )

    # Extract tool call result (guaranteed valid JSON)
    for block in response.content:
        if hasattr(block, "type") and block.type == "tool_use":
            return block.input

    raise ValueError("No tool call in response")
```

Update chapter detector to use structured extraction with fallback:

```python
# workflows/document_processing/nodes/chapter_detector.py

FALLBACK_CHUNK_SIZE = 30000  # Target ~30k words per fallback chunk


def _create_fallback_chunks(markdown: str, word_count: int) -> list[ChapterInfo]:
    """Create pseudo-chapters by splitting into ~30k word chunks.

    Used as fallback when heading-based chapter detection fails.
    Splits on paragraph boundaries to avoid breaking mid-sentence.
    """
    num_chunks = max(1, (word_count + FALLBACK_CHUNK_SIZE - 1) // FALLBACK_CHUNK_SIZE)
    target_chunk_size = len(markdown) // num_chunks

    chunks = []
    current_pos = 0

    for i in range(num_chunks):
        start_pos = current_pos

        if i == num_chunks - 1:
            end_pos = len(markdown)
        else:
            target_pos = start_pos + target_chunk_size
            # Find paragraph break near target
            search_start = max(start_pos, target_pos - 2000)
            search_end = min(len(markdown), target_pos + 2000)
            search_region = markdown[search_start:search_end]
            para_break = search_region.rfind("\n\n")

            if para_break != -1:
                end_pos = search_start + para_break + 2
            else:
                end_pos = target_pos

        chunk_text = markdown[start_pos:end_pos]
        chunks.append(ChapterInfo(
            title=f"Section {i + 1}",
            start_position=start_pos,
            end_position=end_pos,
            author=None,
            word_count=count_words(chunk_text),
        ))
        current_pos = end_pos

    logger.info(f"Created {len(chunks)} fallback chunks (~{FALLBACK_CHUNK_SIZE} words each)")
    return chunks


async def detect_chapters(state: DocumentProcessingState) -> dict[str, Any]:
    # ... heading extraction ...

    schema = {
        "type": "object",
        "properties": {
            "headings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "is_chapter": {"type": "boolean"},
                        "chapter_author": {"type": ["string", "null"]},
                    },
                    "required": ["heading", "is_chapter"],
                },
            },
        },
        "required": ["headings"],
    }

    try:
        # Use structured extraction for guaranteed valid JSON
        result = await extract_structured(
            text=heading_list,
            prompt=prompt,
            schema=schema,
            tier=ModelTier.SONNET,
        )
        analysis = result.get("headings", [])
        chapters = _build_chapter_boundaries(markdown, headings, analysis)

    except Exception as e:
        # Graceful fallback: chunk into ~30k word sections
        logger.warning(f"Chapter detection failed: {e}. Using fallback chunking.")
        chapters = _create_fallback_chunks(markdown, word_count)

    return {"chapters": chapters, "needs_tenth_summary": True}
```

### Step 2: Add Chunking for Large Chapters

Split very long chapters to avoid token limit errors:

```python
# workflows/document_processing/subgraphs/chapter_summarization.py

MAX_CHAPTER_CHARS = 600_000  # 600k chars ≈ 150k tokens, safe for 200k context
CHUNK_SIZE_CHARS = 500_000   # Target chunk size
CHUNK_OVERLAP_CHARS = 2000   # Overlap for context continuity


def _chunk_large_content(content: str) -> list[str]:
    """Split large content into chunks that fit within token limits.

    Uses paragraph boundaries when possible, with overlap for continuity.
    """
    if len(content) <= MAX_CHAPTER_CHARS:
        return [content]

    chunks = []
    current_pos = 0

    while current_pos < len(content):
        end_pos = min(current_pos + CHUNK_SIZE_CHARS, len(content))

        if end_pos < len(content):
            # Find paragraph break near target
            search_start = max(current_pos, end_pos - 5000)
            search_region = content[search_start:end_pos]
            para_break = search_region.rfind("\n\n")

            if para_break != -1:
                end_pos = search_start + para_break + 2
            else:
                # Fall back to word boundary
                while end_pos > current_pos and not content[end_pos].isspace():
                    end_pos -= 1

        chunks.append(content[current_pos:end_pos])

        if end_pos < len(content):
            current_pos = max(current_pos + 1, end_pos - CHUNK_OVERLAP_CHARS)
            while current_pos < len(content) and not content[current_pos].isspace():
                current_pos += 1
        else:
            break

    logger.info(f"Split large chapter into {len(chunks)} chunks")
    return chunks


async def _summarize_single_chapter(chapter, chapter_content, target_words, semaphore):
    async with semaphore:
        chapter_context = f"Chapter: {chapter['title']}"

        # Check if content needs chunking
        chunks = _chunk_large_content(chapter_content)

        if len(chunks) == 1:
            # Normal path - single chunk
            summary = await _summarize_content_chunk(
                content=chapter_content,
                target_words=target_words,
                chapter_context=chapter_context,
            )
        else:
            # Large chapter - summarize each chunk then combine
            logger.info(
                f"Chapter '{chapter['title']}' is too large ({len(chapter_content)} chars), "
                f"splitting into {len(chunks)} chunks"
            )
            chunk_target_words = max(50, target_words // len(chunks))
            chunk_summaries = []

            for i, chunk in enumerate(chunks, 1):
                chunk_summary = await _summarize_content_chunk(
                    content=chunk,
                    target_words=chunk_target_words,
                    chapter_context=chapter_context,
                    chunk_num=i,
                    total_chunks=len(chunks),
                )
                chunk_summaries.append(chunk_summary)

            # Combine chunk summaries
            summary = "\n\n".join(chunk_summaries)

        return {"title": chapter["title"], "author": chapter.get("author"), "summary": summary}
```

### Step 3: Add Memory Monitoring for Marker

Add RAM and GPU memory tracking to Celery tasks:

```python
# services/marker/app/tasks.py

import psutil
import subprocess


def get_memory_stats() -> dict:
    """Get current RAM and GPU memory usage for monitoring."""
    # RAM usage for this process
    ram = psutil.Process().memory_info()
    ram_gb = ram.rss / (1024**3)

    # GPU memory via nvidia-smi
    gpu_used_gb = 0.0
    gpu_total_gb = 0.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_used, gpu_total = map(int, result.stdout.strip().split(", "))
            gpu_used_gb = gpu_used / 1024
            gpu_total_gb = gpu_total / 1024
    except Exception:
        pass

    return {"ram_gb": ram_gb, "gpu_used_gb": gpu_used_gb, "gpu_total_gb": gpu_total_gb}


@celery.task(bind=True)
def convert_document(self, file_path: str, ...) -> dict:
    # Log memory before processing
    before = get_memory_stats()
    logger.info(
        f"[{file_path}] Starting - RAM: {before['ram_gb']:.1f}GB, "
        f"GPU: {before['gpu_used_gb']:.1f}/{before['gpu_total_gb']:.1f}GB"
    )

    try:
        processor = get_processor()
        result = processor.process(file_path)

        # Log memory after processing
        after = get_memory_stats()
        logger.info(
            f"[{file_path}] Complete - RAM: {after['ram_gb']:.1f}GB, "
            f"GPU: {after['gpu_used_gb']:.1f}/{after['gpu_total_gb']:.1f}GB"
        )
        return {"status": "completed", "result": result}

    except Exception as e:
        after = get_memory_stats()
        logger.error(
            f"[{file_path}] Failed - RAM: {after['ram_gb']:.1f}GB, "
            f"GPU: {after['gpu_used_gb']:.1f}/{after['gpu_total_gb']:.1f}GB - Error: {e}"
        )
        return {"status": "failed", "error": str(e)}
```

Reduce batch sizes to 75% for stability:

```yaml
# services/marker/docker-compose.yml
environment:
  - INFERENCE_RAM=24
  - RECOGNITION_BATCH_SIZE=192   # Was 256
  - DETECTOR_BATCH_SIZE=18       # Was 24
  - LAYOUT_BATCH_SIZE=24         # Was 32
  - TABLE_REC_BATCH_SIZE=36      # Was 48
```

### Step 4: Add Relevance Score Tracking

Track relevance scores and filter corpus to top papers:

```python
# workflows/research/subgraphs/academic_lit_review/diffusion_engine.py

async def update_corpus_and_graph(state: DiffusionEngineState) -> dict[str, Any]:
    # ... existing corpus building ...

    for doi in all_relevant_dois:
        paper = candidate_lookup.get(doi) or fallback_papers.get(doi)
        if paper:
            # Ensure papers have relevance scores
            if paper.get("relevance_score") is None:
                # High default for co-citation papers (citation network evidence)
                paper["relevance_score"] = 0.8
            new_corpus_papers[doi] = paper


async def finalize_diffusion(state: DiffusionEngineState) -> dict[str, Any]:
    """Finalize diffusion and filter to top papers by relevance."""
    paper_corpus = state.get("paper_corpus", {})
    max_papers = state["quality_settings"]["max_papers"]

    # Filter to top N papers by relevance score if exceeded max_papers
    if len(paper_corpus) > max_papers:
        sorted_papers = sorted(
            paper_corpus.items(),
            key=lambda x: x[1].get("relevance_score", 0.5),
            reverse=True,
        )
        cutoff_score = sorted_papers[max_papers - 1][1].get("relevance_score", 0.5)
        final_dois = [doi for doi, _ in sorted_papers[:max_papers]]
        logger.info(
            f"Filtered {len(paper_corpus)} papers to {max_papers} "
            f"(relevance cutoff: {cutoff_score:.2f})"
        )
    else:
        final_dois = list(paper_corpus.keys())

    return {"final_corpus_dois": final_dois}
```

## Prevention

### Structured Output Guidelines

1. **Use tool use for JSON extraction** - Anthropic tool_choice="required" guarantees valid JSON
2. **Define explicit schemas** with all required fields and types
3. **Add graceful fallbacks** for LLM failures (chunking, default values)
4. **Avoid prompt-based JSON** - schema hints are not enforced

### Large Content Guidelines

1. **Check content size before processing** - 600k chars ≈ 150k tokens
2. **Split at natural boundaries** - paragraphs > sentences > words
3. **Add overlap between chunks** - 2k chars maintains context
4. **Combine chunk results** appropriately for the task

### Memory Monitoring Guidelines

1. **Log memory before/after processing** - identifies leaks and pressure
2. **Use nvidia-smi for GPU memory** - subprocess with timeout
3. **Reduce batch sizes under pressure** - 75% provides stability margin
4. **Track OOM patterns** - correlate with document sizes

### Relevance Filtering Guidelines

1. **Track relevance scores on all papers** - enables quality filtering
2. **Set sensible defaults** - 0.8 for citation-discovered, 0.5 for unknown
3. **Filter at finalization** - not during discovery
4. **Log cutoff scores** - helps tune quality settings

## Files Modified

- `workflows/shared/llm_utils.py`: Added `extract_structured()` with tool use
- `workflows/document_processing/nodes/chapter_detector.py`: Structured output, fallback chunking
- `workflows/document_processing/subgraphs/chapter_summarization.py`: Large chapter chunking
- `services/marker/app/tasks.py`: Memory monitoring, before/after logging
- `services/marker/docker-compose.yml`: Reduced batch sizes
- `workflows/research/subgraphs/academic_lit_review/diffusion_engine.py`: Relevance scores, top-N filtering

## Related Patterns

- [Citation Network Academic Review Workflow](../../patterns/langgraph/citation-network-academic-review-workflow.md) - Quality presets and paper limits
- [Anthropic Claude Integration with Extended Thinking](../../patterns/llm-interaction/anthropic-claude-integration-extended-thinking.md) - Tool use patterns

## Related Solutions

- [Academic Literature Review Reliability Fixes](./academic-lit-review-reliability-fixes.md) - Pydantic structured output
- [Paper Acquisition Robustness](./paper-acquisition-robustness.md) - Pipeline reliability

## References

- [Anthropic Tool Use Documentation](https://docs.anthropic.com/en/docs/tool-use)
- [psutil Process Memory](https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info)
