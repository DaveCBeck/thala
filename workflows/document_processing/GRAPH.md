# Document Processing Graph

## Overview
Main LangGraph workflow that orchestrates document processing from input to final storage.

## Workflow

```
START
  ↓
resolve_input
  ↓
create_zotero_stub
  ↓
[route_by_source_type]
  ├─ markdown → smart_chunker
  └─ needs_marker → process_marker (with retry)
      ↓
update_store
  ↓
[fan_out_to_agents] (parallel)
  ├─ generate_summary
  └─ check_metadata
      ↓
save_short_summary
  ↓
update_zotero
  ↓
detect_chapters
  ↓
[route_by_doc_size]
  ├─ needs_tenth → chapter_summarization (subgraph)
  │                  ↓
  │                save_tenth_summary
  └─ skip_tenth (short docs)
      ↓
finalize
  ↓
END
```

## Key Features

### Conditional Routing
- **route_by_source_type**: Routes markdown files directly to chunker, others to Marker
- **route_by_doc_size**: Only runs 10:1 summary for long documents (>50k words)

### Parallel Execution
- **fan_out_to_agents**: Runs summary and metadata extraction in parallel using Send()

### Retry Policy
- **process_marker**: Retries up to 3 times with exponential backoff (2.0x)

### Subgraphs
- **chapter_summarization**: Map-reduce pattern for parallel chapter summarization

## State Management

### Accumulation
- `store_records`: Accumulated list (operator.add)
- `errors`: Accumulated list (operator.add)
- `chapter_summaries`: Accumulated list (operator.add)
- `metadata_updates`: Merged dict (merge_metadata reducer)

### Flags
- `is_already_markdown`: Controls routing to chunker vs Marker
- `needs_tenth_summary`: Controls 10:1 summary generation

## Usage

```python
from workflows.document_processing import process_document

result = await process_document(
    source="/path/to/document.pdf",
    title="Optional Title",
)
```
