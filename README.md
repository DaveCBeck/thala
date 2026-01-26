# Thala

A personal knowledge system that goes beyond storage—it thinks alongside you.

For a decade I've run this as a done-for-you service: acting as a "second brain" for clients, knowing their thought processes and beliefs well enough to have a "mini-client" running in my subconscious. This project attempts to automate that relationship, while throwing off some new research material for me to read.

**The goal:** Give technical users who put the time in their own second brain—with background processes that mirror your own default mode network. It maintains coherence, recognizes patterns, and develops genuine context about *you* over time.

**Target audience:** Self-hosters comfortable with Docker. Particularly relevant if you use Obsidian, Roam Research, or other connected thinking tools and want something that actively processes rather than just stores.

---

> **Early Development**
>
> Currently building out the research workflow layer—the "executive" functions that gather and synthesize information. The innovative parts (coherence layer, background processing, the actual "second brain" behavior) come later.
>
> Documentation, setup guides, and a proper release will follow once there's something meaningful to run.

---

## Vision

```
┌─────────────────────────────────────────────────────────────────┐
│ Executive (conscious)                                           │
│   Directed research, writing, decisions—user-initiated tasks    │
├─────────────────────────────────────────────────────────────────┤
│ Subconscious (background)                                       │
│   Coherence maintenance, pattern recognition, goal alignment    │
│   Runs when idle—default mode network                           │
├─────────────────────────────────────────────────────────────────┤
│ Stores                                                          │
│   top_of_mind │ coherence │ who_i_was │ store │ forgotten_store │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Workflows Layer                              │
│  LangGraph state machines orchestrating multi-step research tasks      │
├────────────────────────────────────────────────────────────────────────┤
│                          LangChain Tools                               │
│  Structured tool interfaces (OpenAlex, web scraping, academic search)  │
├────────────────────────────────────────────────────────────────────────┤
│                             Core Layer                                 │
│  Stores (ES/Chroma/Zotero) │ Images │ Scraping │ Task Queue │ Utils   │
├────────────────────────────────────────────────────────────────────────┤
│                          Services Layer                                │
│  Docker: Elasticsearch │ Chroma │ Zotero │ Firecrawl │ Retrieve-Academic│
└────────────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
thala/
├── core/                          # Foundation layer
│   ├── stores/                    # Data persistence (ES, Chroma, Zotero)
│   ├── images/                    # Diagram generation providers
│   ├── scraping/                  # Web scraping abstractions
│   ├── task_queue/                # Background task infrastructure
│   └── utils/                     # Async helpers, caching, context managers
│
├── workflows/                     # LangGraph workflow orchestration
│   ├── research/                  # Research workflows
│   │   ├── academic_lit_review/   # PhD-level literature reviews
│   │   ├── web_research/          # Multi-source web research
│   │   └── book_finding/          # Book discovery and sourcing
│   ├── enhance/                   # Content enhancement pipeline
│   │   ├── supervision/           # Human-in-the-loop review
│   │   ├── editing/               # Style and clarity editing
│   │   └── fact_check/            # Claim verification
│   ├── output/                    # Content generation
│   │   ├── evening_reads/         # Substack series generation
│   │   └── illustrate/            # Document diagram generation
│   ├── wrappers/                  # Workflow composition
│   │   ├── synthesis/             # Multi-workflow synthesis
│   │   └── multi_lang/            # Cross-language research
│   └── shared/                    # Shared utilities across workflows
│
├── langchain_tools/               # LangChain tool interfaces
│   └── openalex/                  # Academic search (OpenAlex API)
│
├── services/                      # Docker service infrastructure
│   ├── firecrawl/                 # Local web scraping service
│   ├── zotero/                    # Zotero CRUD API plugin
│   └── retrieve-academic/         # Academic document retrieval
│
├── mcp_server/                    # MCP server interface
├── testing/                       # Test utilities and fixtures
└── scripts/                       # Operational scripts
```

## Module Documentation

| Module | Purpose | Docs |
|--------|---------|------|
| **Core** |  |  |
| `core/` | Foundation layer overview | [README](core/README.md) |
| `core/stores/` | Data persistence (ES, Chroma, Zotero) | [README](core/stores/README.md) |
| `core/images/` | Diagram generation providers | [README](core/images/README.md) |
| `core/scraping/` | Web scraping abstractions | [README](core/scraping/README.md) |
| `core/task_queue/` | Background task infrastructure | [README](core/task_queue/README.md) |
| `core/utils/` | Async helpers, caching, context managers | [README](core/utils/README.md) |
| **Workflows** |  |  |
| `workflows/` | Workflow layer overview | [README](workflows/README.md) |
| `workflows/research/academic_lit_review/` | PhD-level literature reviews | [README](workflows/research/academic_lit_review/README.md) |
| `workflows/research/web_research/` | Multi-source web research | [README](workflows/research/web_research/README.md) |
| `workflows/research/book_finding/` | Book discovery and sourcing | [README](workflows/research/book_finding/README.md) |
| `workflows/enhance/` | Content enhancement pipeline | [README](workflows/enhance/README.md) |
| `workflows/enhance/supervision/` | Human-in-the-loop review | [README](workflows/enhance/supervision/README.md) |
| `workflows/enhance/editing/` | Style and clarity editing | [README](workflows/enhance/editing/README.md) |
| `workflows/enhance/fact_check/` | Claim verification | [README](workflows/enhance/fact_check/README.md) |
| `workflows/output/evening_reads/` | Substack series generation | [README](workflows/output/evening_reads/README.md) |
| `workflows/output/illustrate/` | Document diagram generation | [README](workflows/output/illustrate/README.md) |
| `workflows/wrappers/synthesis/` | Multi-workflow synthesis | [README](workflows/wrappers/synthesis/README.md) |
| `workflows/wrappers/multi_lang/` | Cross-language research | [README](workflows/wrappers/multi_lang/README.md) |
| `workflows/shared/` | Shared utilities | [README](workflows/shared/README.md) |
| **Tools & Services** |  |  |
| `langchain_tools/` | LangChain tool interfaces | [README](langchain_tools/README.md) |
| `langchain_tools/openalex/` | OpenAlex academic search | [README](langchain_tools/openalex/README.md) |
| `services/` | Docker service infrastructure | [README](services/README.md) |
| `services/firecrawl/` | Local web scraping service | [README](services/firecrawl/README.md) |
| `services/zotero/zotero-local-crud/` | Zotero CRUD API plugin | [README](services/zotero/zotero-local-crud/README.md) |
