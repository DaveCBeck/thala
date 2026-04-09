# Thala

ALWAYS read and understand relevant files before proposing code edits. Do not speculate about code you have not inspected. If the user references a specific file/path, you MUST open and inspect it before explaining or proposing fixes. Be rigorous and persistent in searching code for key facts. Thoroughly review the style, conventions, and abstractions of the codebase before implementing new features or abstractions.

The year is 2026.

## Services (Docker)

- **Zotero** - Headless via Xvfb/Selkies, debug-bridge:23119, local API
- **Marker** - PDF/DOCX/EPUB extraction via FastAPI, watch folder support
- **Translation-server** - URL → CSL-JSON metadata
- **Elasticsearch** - Primary document stores
- **Chroma** - Vector store for rapid search

## Python - always use the virtual environment at `.venv/`.

## Hardware Constraints

```
CPU:  AMD Ryzen 9 5950X
GPU:  RTX 4060 Ti 16GB VRAM
RAM:  64GB
Env:  WSL2, Docker monorepo
```

GPU shared across: Marker/Surya, embeddings, local LLM inference. Queue heavy tasks.

## Environment Variables

All environment variables are managed via `.env` (gitignored) with `.env.example` as the template.

## Documentation

**Read relevant docs before coding. Update them when making changes.**

| Doc | Purpose |
|-----|---------|
| [docs/architecture.md](docs/architecture.md) | System architecture and data flow |
| [docs/patterns.md](docs/patterns.md) | Code patterns and conventions |
| [docs/stack.md](docs/stack.md) | Technology versions and constraints |
| [workflows/README.md](workflows/README.md) | Research workflow documentation |
| [core/README.md](core/README.md) | Stores, embedding, scraping |
| [services/README.md](services/README.md) | Docker services |
| [langchain_tools/README.md](langchain_tools/README.md) | LangChain tool interfaces |
| [tests/README.md](tests/README.md) | Test patterns and utilities |

## Testing

**ALWAYS read `tests/README.md` before running or creating tests.**

## Working Style

- User dictates via voice transcription—ask if unclear
- Terse, concise code
- Inline comments only where logic isn't self-evident
- Don't create new documentation files unless explicitly asked
- Type hints for public interfaces
- Minimal abstraction—prefer explicit over clever
- Always update `.env` and `.env.example` when adding environment variables


## Sensitive References

**Do not mention "Anna's Archive" in committed code or documentation.** The book search functionality uses a private submodule for source configuration. References in dev.md, claude.md, and the private submodule are fine - just not in committed public code.

## Agents vs Commands

**Agents**: Specific workflows, auto-delegated or explicit. Isolated context. Use for orchestration or specialized expertise.

**Commands**: User-invoked repeatable prompts. Use for research pipelines that orchestrate multiple sub-agents—maximizes Claude Code Max subscription value.
