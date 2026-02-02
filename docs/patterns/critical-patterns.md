# Critical Patterns - Required Reading

**Purpose:** High-value patterns that represent proven approaches in the Thala codebase. These are the "how to do X the right way" patterns that should be followed consistently.

**Format:** Each entry summarizes a pattern with links to full documentation.

---

## How to Add Patterns

When a pattern has been:
- Verified in production (`verified_in_production: true`)
- Used across multiple modules
- Proven to solve a recurring need elegantly

Add a summary entry here with a link to the full pattern documentation.

---

## Patterns by Category

### LangGraph

- **[Pipeline Simplification with Bridge Compatibility](./langgraph/pipeline-simplification-with-bridge-compatibility.md)** - Replace complex multi-phase iteration loops with simpler linear pipelines while preserving downstream compatibility through a bridge node. *Production-verified.*
- **[Unified Quality Tier System](./langgraph/unified-quality-tier-system.md)** - Centralized quality configuration propagated through workflows with consistent presets. *Production-verified.*
- **[Citation Network Academic Review Workflow](./langgraph/citation-network-academic-review-workflow.md)** - Multi-phase literature review with citation graph analysis and BERTopic clustering. *Production-verified.*
- **[Synthesis Workflow Orchestration](./langgraph/synthesis-workflow-orchestration.md)** - Parallel research workers with Send() pattern and reducer-based aggregation. *Production-verified.*

### LLM Interaction

- **[LangChain Tools for Store Integration](./llm-interaction/langchain-tools-store-integration.md)** - Wrapping internal stores (ES, Chroma, Zotero) for LangChain agent consumption with lazy initialization and Pydantic schemas. *Production-verified.*
- **[LLM Factory Pattern](./llm-interaction/llm-factory-pattern.md)** - Centralized model instantiation with tier-based selection and configuration. *Production-verified.*
- **[DeepSeek Integration Patterns](./llm-interaction/deepseek-integration-patterns.md)** - Cost-optimized LLM calls with DeepSeek V3/R1 integration and automatic cache warmup. *Production-verified.*
- **[Batch API Cost Optimization](./llm-interaction/batch-api-cost-optimization.md)** - 50% cost reduction using Anthropic Batch API with LangSmith tracing. *Production-verified.*

### Async Python

- **[Streaming Producer-Consumer Pipeline](./async-python/streaming-producer-consumer-pipeline.md)** - Bounded queues with backpressure for GPU-intensive processing. *Production-verified.*
- **[Concurrent Scraping with TTL Cache](./async-python/concurrent-scraping-with-ttl-cache.md)** - Parallel web scraping with rate limiting and deterministic ordering. *Production-verified.*

### Data Pipeline

- **[Unified Content Retrieval Pipeline](./data-pipeline/unified-content-retrieval-pipeline.md)** - 5-stage content acquisition with DOI detection, classification, and fallback chains. *Production-verified.*
- **[Two-Stage Pipeline Architecture](./data-pipeline/two-stage-pipeline-architecture.md)** - Bounded PDF queue + unbounded LLM queue for memory-safe processing. *Production-verified.*
- **[Multi-Workflow Task Queue](./data-pipeline/multi-workflow-task-queue.md)** - Registry-based workflow dispatch with budget tracking and checkpointing. *Production-verified.*
- **[Incremental Checkpointing for Iterative Workflows](./data-pipeline/incremental-checkpointing-iterative-workflows.md)** - Mid-iteration resumption with gzip-compressed checkpoints and graceful shutdown. *Production-verified.*

### Observability

- **[Comprehensive LangSmith Tracing](./observability/comprehensive-langsmith-tracing.md)** - End-to-end tracing with tool wrappers, metadata propagation, and tag standardization. *Production-verified.*
- **[Centralized Logging Configuration](./observability/centralized-logging-configuration.md)** - Two-tier logging with third-party logger segregation. *Production-verified.*

### Stores (ES/Chroma/Zotero)

- **[Mandatory Archive Before Delete](./stores/mandatory-archive-before-delete.md)** - All destructive operations archive to history stores first. *Production-verified.*
- **[Compression Level Index Routing](./stores/compression-level-index-routing.md)** - L0/L1/L2 index strategy for full text vs summaries. *Production-verified.*

---

## Recently Added

- **2026-01-30**: [Incremental Checkpointing for Iterative Workflows](./data-pipeline/incremental-checkpointing-iterative-workflows.md) - Mid-iteration resumption for expensive operations
- **2026-01-29**: Updated critical patterns index with 17 production-verified patterns from audit
- **2026-01-28**: [Pipeline Simplification with Bridge Compatibility](./langgraph/pipeline-simplification-with-bridge-compatibility.md) - From commit 45a630c
- **2026-01-28**: [Comprehensive LangSmith Tracing](./observability/comprehensive-langsmith-tracing.md) - Full tracing infrastructure
- **2026-01-28**: [DeepSeek Integration Patterns](./llm-interaction/deepseek-integration-patterns.md) - Cost optimization with DeepSeek
- **2026-01-26**: [LLM Factory Pattern](./llm-interaction/llm-factory-pattern.md) - Centralized model instantiation
- **2025-12-17**: [LangChain Tools for Store Integration](./llm-interaction/langchain-tools-store-integration.md) - From commit 42e478d
