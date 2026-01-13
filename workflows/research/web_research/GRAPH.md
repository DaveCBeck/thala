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
  │  web_researcher (up to 3)    │   │
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
save_findings
  ↓
END
```

## Web Researcher Subgraph

The main workflow uses web researchers (up to 3 concurrent per iteration):

```
generate_queries → execute_searches → scrape_pages (3) → compress_findings
```
- Sources: Firecrawl, Perplexity (parallel)
- Prompt: Emphasizes recency, domain authority, bias detection

## Standalone Workflows

For academic papers and books, use the dedicated standalone workflows:

### Academic Literature Review
```python
from workflows.academic_lit_review import academic_lit_review

result = await academic_lit_review(
    topic="machine learning in drug discovery",
    research_questions=["How are GNNs used for molecular property prediction?"],
    quality="standard",  # test | quick | standard | comprehensive | high_quality
    language="en",       # ISO 639-1 code (29 languages supported)
)
```

### Book Finding
```python
from workflows.book_finding import book_finding

result = await book_finding(
    theme="organizational resilience",
    brief="Focus on practical approaches",  # optional
    language="es",                          # ISO 639-1 code
)
```
Generates 9 book recommendations across three categories (analogous domain, inspiring action, expressive fiction) using Opus, then searches and processes them.

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
- **Web researchers**: Up to 3 concurrent web researchers per iteration
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

| Source | Type | Used In |
|--------|------|---------|
| Firecrawl | General web | deep_research |
| Perplexity | AI-synthesized | deep_research |
| OpenAlex | Academic papers | academic_lit_review |
| book_search | Books | book_finding |

## Usage

```python
from workflows.web_research import deep_research

# Standard English research
result = await deep_research(
    query="Impact of AI on software engineering jobs",
    quality="standard",  # test | quick | standard | comprehensive | high_quality
)

# Research in Spanish (workflow runs entirely in Spanish)
result = await deep_research(
    query="impacto de IA en empleos",
    language="es",
)

# For academic research, use the standalone workflow
from workflows.academic_lit_review import academic_lit_review
result = await academic_lit_review(
    topic="postcolonial literary theory",
    research_questions=["How has Said's Orientalism influenced the field?"],
)

print(result["final_report"])
print(f"Sources: {len(result['citations'])}")
```

## Output

```python
{
    "final_report": str,           # Comprehensive markdown report (in specified language)
    "status": str,                 # "success", "partial", or "failed"
    "langsmith_run_id": str,       # LangSmith trace ID for debugging
    "errors": list[dict],          # Any errors encountered
    "source_count": int,           # Number of sources used
    "started_at": datetime,        # Workflow start time
    "completed_at": datetime,      # Workflow end time
}
```
