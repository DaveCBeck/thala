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
  │  [fan_out] (parallel)        │   │
  │    ├─ web_researcher         │   │
  │    ├─ academic_researcher    │   │
  │    └─ book_researcher        │   │
  │         ↓                    │   │
  │  aggregate_findings ─────────┼───┘
  ├─ refine_draft ───────────────┘
  │
  └─ research_complete
       ↓
final_report (OPUS)
  ↓
process_citations
  ↓
translate_report (if translate_to)
  ↓
save_findings
  ↓
END
```

## Specialized Researcher Subgraphs

Three specialized researchers run in parallel per iteration (default: 1 of each):

### Web Researcher (Firecrawl + Perplexity)
```
generate_queries → execute_searches → scrape_pages (3) → compress_findings
```
- Sources: Firecrawl, Perplexity (parallel)
- Prompt: Emphasizes recency, domain authority, bias detection

### Academic Researcher (OpenAlex)
```
generate_queries → execute_searches → scrape_pages (3) → compress_findings
```
- Sources: OpenAlex (scholarly literature)
- Prompt: Emphasizes peer-review, citations, methodology
- Sorted by citation count

### Book Researcher (book_search)
```
generate_queries → execute_searches → scrape_pages (1) → compress_findings
```
- Sources: book_search
- Prompt: Emphasizes author credentials, publisher reputation
- Prioritizes PDF format

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
- **Specialized researchers**: 3 concurrent researcher types via Send() (web, academic, book)
- **Default allocation**: 1 web + 1 academic + 1 book per iteration (configurable)
- **Parallel scraping**: Top results scraped concurrently with TTL cache (1hr) to avoid redundant fetches
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

# Standard English research
result = await deep_research(
    query="Impact of AI on software engineering jobs",
    depth="standard",  # quick | standard | comprehensive
    max_sources=20,
)

# Research in Spanish
result = await deep_research(
    query="impacto de IA en empleos",
    language="es",
)

# Research in Spanish, translate to English
result = await deep_research(
    query="impacto de IA en empleos",
    language="es",
    translate_to="en",
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
    "translated_report": str,      # If translate_to was specified
}
```
