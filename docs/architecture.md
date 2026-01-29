# Architecture

Thala is a personal knowledge system with three layers: conscious (user-initiated), subconscious (background), and infrastructure (services).

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Executive (conscious)                                           │
│   workflows/ - Research, writing, decisions (user-initiated)    │
│   langchain_tools/ - Tool interfaces for agents                 │
├─────────────────────────────────────────────────────────────────┤
│ Subconscious (background)                                       │
│   Coherence maintenance, pattern recognition, goal alignment    │
│   Runs when idle—default mode network                           │
├─────────────────────────────────────────────────────────────────┤
│ Core Infrastructure                                             │
│   core/ - Stores, embedding, scraping, configuration            │
│   services/ - Docker services (ES, Chroma, Zotero, etc.)        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Query / Input
    ↓
[Workflow Entry] (e.g., academic_lit_review, web_research)
    ↓
[State Initialization] → build_initial_state(input, quality_settings)
    ↓
[LangGraph Execution]
    ├→ Discovery: Query stores, generate searches, execute APIs
    ├→ Processing: PDF extraction, summarization, metadata
    ├→ Analysis: Clustering, relevance scoring, validation
    └→ Synthesis: Structured output, citations, reports
    ↓
[Store Integration]
    ├→ Elasticsearch: Main results
    ├→ Zotero: Citations (source of truth)
    ├→ Chroma: Embeddings
    └→ MCP Server: Tool availability
    ↓
Return to User / Agent
```

## Component Relationships

| Layer | Component | Depends On | Provides |
|-------|-----------|------------|----------|
| **Workflows** | academic_lit_review | core, shared | Literature reviews |
| **Workflows** | web_research | core, shared, langchain_tools | Research reports |
| **Workflows** | book_finding | shared | Book recommendations |
| **Workflows** | supervised_lit_review | academic_lit_review | Enhanced reviews |
| **LangChain** | deep_research | workflows/web_research | Tool wrapper |
| **LangChain** | search_memory | core/stores | Memory search |
| **Core** | stores/ | services | Data persistence |
| **Core** | embedding.py | OpenAI/Ollama | Embeddings |
| **Core** | scraping/ | Firecrawl/Playwright | Web content |
| **Shared** | llm_utils/ | Anthropic API | Structured output |

## Store Architecture

**Zotero is source of truth for full text.** All other stores hold metadata, embeddings, and processed derivatives. Every record includes `zotero_key` for cross-reference.

```
┌──────────────────┐    ┌──────────────────┐
│ Elasticsearch    │    │ Elasticsearch    │
│ (coherence:9201) │    │ (forgotten:9200) │
├──────────────────┤    ├──────────────────┤
│ • coherence      │    │ • who_i_was      │
│ • store          │    │ • forgotten_store│
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────┴──────┐
              │   Chroma    │
              │  (:8000)    │
              ├─────────────┤
              │ top_of_mind │
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │   Zotero    │
              │  (:23119)   │
              ├─────────────┤
              │ Full text   │
              │ Citations   │
              └─────────────┘
```

## Workflow Composition

Workflows use LangGraph's StateGraph with composable subgraphs:

```python
# Example: Academic lit review composition
graph = StateGraph(AcademicLitReviewState)
graph.add_node("keyword_search", keyword_search_subgraph)
graph.add_node("citation_expansion", citation_network_subgraph)
graph.add_node("process_papers", paper_processing_subgraph)
graph.add_node("clustering", clustering_subgraph)
graph.add_node("synthesis", synthesis_subgraph)
graph.add_edge("keyword_search", "citation_expansion")
# ... routing logic
```

## Model Tiers

Three tiers for cost/quality trade-offs:

| Tier | Model | Use Case |
|------|-------|----------|
| `HAIKU` | claude-3-5-haiku | Fast, low-cost (filtering, classification) |
| `SONNET` | claude-sonnet-4 | Balanced (default for most tasks) |
| `OPUS` | claude-opus-4 | Maximum quality (synthesis, complex analysis) |

Opus supports extended thinking for complex reasoning tasks.

## Environment Configuration

All services configurable via environment variables with sensible defaults:

```bash
# Mode
THALA_MODE=dev              # Enable LangSmith tracing

# Stores
THALA_ES_COHERENCE_HOST=http://localhost:9201
THALA_ES_FORGOTTEN_HOST=http://localhost:9200
THALA_CHROMA_HOST=localhost
THALA_CHROMA_PORT=8000
THALA_ZOTERO_HOST=localhost
THALA_ZOTERO_PORT=23119

# Caching
THALA_CACHE_DIR=/home/dave/thala/.cache
THALA_CACHE_DISABLED=false
```
