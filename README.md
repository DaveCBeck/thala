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

## Project Structure

| Directory | Purpose | Docs |
|-----------|---------|------|
| `workflows/` | LangGraph research & processing workflows | [README](workflows/README.md) |
| `core/` | Stores, embedding, scraping foundations | [README](core/README.md) |
| `services/` | Docker services (Elasticsearch, Chroma, Zotero) | [README](services/README.md) |
| `langchain_tools/` | LangChain tool interfaces | [README](langchain_tools/README.md) |
