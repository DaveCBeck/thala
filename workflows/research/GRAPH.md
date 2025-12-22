# Deep Research Graph

## Overview
LangGraph workflow implementing a Self-Balancing Diffusion Algorithm for comprehensive research with memory integration, parallel researcher agents, and iterative refinement.

## Main Workflow

```
START
  ↓
clarify_intent
  ↓
[route_after_clarify]
  ↓
create_brief
  ↓
search_memory
  ↓
iterate_plan (OPUS)
  ↓
supervisor (OPUS) ←──────────────────┐
  ↓                                  │
[route_supervisor_action]            │
  ├─ conduct_research ───────────┐   │
  │    ↓                         │   │
  │  [fan_out] (parallel, max 3) │   │
  │    ├─ researcher             │   │
  │    ├─ researcher             │   │
  │    └─ researcher             │   │
  │         ↓                    │   │
  │  aggregate_findings ─────────┼───┘
  │                              │
  ├─ refine_draft ───────────────┘
  │
  └─ research_complete
       ↓
final_report (OPUS)
  ↓
process_citations
  ↓
save_findings
  ↓
END
```

## Researcher Subgraph

Each researcher agent runs this subgraph independently:

```
START
  ↓
generate_queries (HAIKU)
  ↓
execute_searches
  ├─ Firecrawl    (web)       ─┐
  ├─ Perplexity   (AI search) ─┼─ parallel
  ├─ OpenAlex     (academic)  ─┤
  └─ Books        (books)     ─┘
       ↓
  [deduplicate by URL]
       ↓
scrape_pages (top 3)
  ↓
compress_findings (SONNET)
  ↓
END
```

## Key Features

### Self-Balancing Diffusion Algorithm
- **Supervisor loop**: Iteratively generates questions, launches researchers, refines draft
- **Completeness tracking**: Scores 0-1 based on areas explored vs. remaining
- **Dynamic stopping**: Completes when sufficient or max iterations reached
- **Depth presets**: quick (2 iter), standard (4 iter), comprehensive (8 iter)

### Memory-First Research
- **search_memory**: Queries Thala stores before web search
- **iterate_plan**: Customizes research plan based on existing knowledge (OPUS)
- **Avoids redundancy**: Skips topics already well-covered in memory

### Parallel Execution
- **researcher fan-out**: Up to 3 concurrent researcher agents via Send()
- **4-source search**: Firecrawl/Perplexity/OpenAlex/Books run in parallel per query
- **Finding aggregation**: Results merged via Annotated[list, add] reducer

### Retry Policies
- **supervisor**: 3 attempts with 2.0x backoff
- **final_report**: 2 attempts with 2.0x backoff
- **process_citations**: 2 attempts with 2.0x backoff

## State Management

### Accumulation (operator.add)
- `research_findings`: Aggregated from all researcher agents
- `supervisor_messages`: LangGraph message history
- `errors`: Error tracking across all nodes

### Diffusion State
- `iteration`: Current loop count
- `max_iterations`: Depth-based limit
- `completeness_score`: 0-1 estimated coverage (auto-completes at 85%)
- `areas_explored`: Topics already researched
- `areas_to_explore`: Remaining topics
- `last_decision`: Last supervisor action for debugging

### Routing Flags
- `current_status`: Controls supervisor routing
  - `conduct_research` → fan-out to researchers
  - `refine_draft` → update draft with findings
  - `research_complete` → proceed to final report

## Search Sources

| Source | Type | Metadata |
|--------|------|----------|
| Firecrawl | General web | None |
| Perplexity | AI-synthesized | None |
| OpenAlex | Academic papers | DOI, authors, citations, OA status |
| Books | Book search | MD5, format, publisher, size |

## Usage

```python
from workflows.research import deep_research

result = await deep_research(
    query="Impact of AI on software engineering jobs",
    depth="standard",  # quick | standard | comprehensive
    max_sources=20,
)

print(result["final_report"])
print(f"Sources: {len(result['citations'])}")
```

## Output

```python
{
    "final_report": str,           # Comprehensive markdown report
    "citations": list[dict],       # Structured citation data
    "citation_keys": list[str],    # Zotero keys if integrated
    "store_record_id": str,        # UUID in coherence store
    "research_findings": list,     # Raw findings from researchers
    "diffusion": {
        "iteration": int,
        "completeness_score": float,
    },
}
```
