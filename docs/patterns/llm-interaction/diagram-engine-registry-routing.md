---
name: diagram-engine-registry-routing
title: Diagram Engine Registry and Subtype-Based Routing
date: 2026-02-08
category: llm-interaction
applicability:
  - "Systems with multiple rendering backends where availability varies by deployment"
  - "Workflows that need to route to the best tool for a specific task variant"
  - "Graceful degradation when optional dependencies are missing"
components: [registry, routing, async_task]
complexity: simple
verified_in_production: true
tags: [mermaid, graphviz, svg, registry, routing, fallback, lazy-initialization, diagram]
---

# Diagram Engine Registry and Subtype-Based Routing

## Intent

Detect available diagram rendering engines once at startup, then route diagram generation requests to the best engine for each diagram subtype — with automatic fallback to a universally available engine.

## Problem

1. **Optional dependencies**: Mermaid (`mmdc`) and Graphviz (`dot`) may not be installed in all environments
2. **Per-call checks are wasteful**: Checking binary availability on every diagram request adds latency
3. **Different engines suit different diagrams**: Flowcharts render better in Mermaid; hierarchy diagrams suit Graphviz
4. **Hard failures are unacceptable**: A missing engine shouldn't crash the workflow

## Solution

Two components:

1. **Registry** (`registry.py`): Lazy singleton that checks engine availability once, caches results
2. **Routing logic** (in workflow node): Maps diagram subtypes to preferred engines, falls back to SVG

### Registry

```python
# workflows/shared/diagram_utils/registry.py

_available_engines: set[str] = set()
_checked = False

def get_available_engines() -> set[str]:
    """Return set of available diagram engines. Checked once."""
    global _checked
    if not _checked:
        _available_engines.add("svg")  # always available (cairosvg)

        try:
            import mmdc  # noqa: F401
            _available_engines.add("mermaid")
        except ImportError:
            logger.info("Mermaid engine unavailable (mmdc not installed)")

        if shutil.which("dot"):
            _available_engines.add("graphviz")
        else:
            logger.info("Graphviz engine unavailable (dot binary not found)")

        _checked = True
    return _available_engines

def is_engine_available(engine: str) -> bool:
    return engine in get_available_engines()

def reset_registry() -> None:
    """Reset for testing."""
    global _checked
    _available_engines.clear()
    _checked = False
```

### Routing Logic

```python
# workflows/output/illustrate/nodes/generate_additional.py

_MERMAID_SUBTYPES = {"flowchart", "sequence", "concept_map"}
_GRAPHVIZ_SUBTYPES = {"network_graph", "hierarchy", "dependency_tree"}

async def _generate_diagram(location_id, plan, brief, config):
    subtype = plan.diagram_subtype
    result = None

    if subtype in _MERMAID_SUBTYPES and is_engine_available("mermaid"):
        result = await generate_mermaid_with_selection(...)
    elif subtype in _GRAPHVIZ_SUBTYPES and is_engine_available("graphviz"):
        result = await generate_graphviz_with_selection(...)

    # Fallback to SVG if preferred engine failed or unavailable
    if result is None or not result.success:
        result = await generate_diagram(...)  # Raw SVG pipeline
```

### Fallback Chain

```
Preferred engine (Mermaid/Graphviz) based on subtype
         |
         | (unavailable or generation failed)
         v
Raw SVG pipeline (always available via cairosvg)
         |
         | (failed)
         v
DiagramResult(success=False, error="...")
```

## Key Design Decisions

### Lazy Singleton, Not Per-Call

The registry checks availability once at first access. This avoids:
- Repeated `shutil.which("dot")` system calls
- Repeated `import mmdc` attempts
- Race conditions in concurrent workflows

### Routing in Workflow Node, Not Engine

The routing decision lives in `generate_additional.py`, not in a generic dispatcher. This gives the workflow node:
- Full control over logging (`"Routing diagram {id} to Mermaid engine"`)
- Ability to pass engine-specific configuration
- Clear fallback path visible in one function

### SVG Always Available

The `svg` engine (raw SVG via LLM + cairosvg conversion) is always registered. It serves as the universal fallback, ensuring diagram generation never fails due to missing optional dependencies.

### Test Reset

`reset_registry()` allows tests to manipulate availability without module-level side effects.

## Guidelines

| Subtype | Preferred Engine | Rationale |
|---------|-----------------|-----------|
| `flowchart` | Mermaid | Native flowchart syntax |
| `sequence` | Mermaid | Sequence diagram support |
| `concept_map` | Mermaid | Subgraph grouping |
| `network_graph` | Graphviz | Force-directed layout |
| `hierarchy` | Graphviz | Tree layout algorithms |
| `dependency_tree` | Graphviz | Hierarchical rendering |
| `custom_artistic` | SVG | Full creative control |
| (unknown/None) | SVG | Safe default |

## Known Uses

- `workflows/shared/diagram_utils/registry.py` — Engine availability registry
- `workflows/output/illustrate/nodes/generate_additional.py` — Subtype-based routing
- `workflows/shared/diagram_utils/__init__.py` — Re-exports registry functions

## Consequences

### Benefits

- **Zero-cost availability checks** after first call
- **Graceful degradation**: Missing engines don't crash workflows
- **Clear routing**: Subtype-to-engine mapping is explicit and auditable
- **Testable**: `reset_registry()` enables isolated testing

### Trade-offs

- **Static mapping**: New diagram subtypes require updating the routing sets
- **No dynamic scoring**: Doesn't consider engine quality per-diagram, just type
- **Global state**: Registry is module-level; `reset_registry()` needed for test isolation

## Related Patterns

- [Validate-Repair-Render Loop](./validate-repair-render-loop.md) — What happens inside each engine
- [Parallel Candidate Vision Selection](./parallel-candidate-vision-selection.md) — `generate_*_with_selection()` uses multi-candidate + vision
- [Document Illustration Workflow](../langgraph/document-illustration-workflow.md) — Orchestrates diagram generation

## References

- Commit: `b5336d9` — feat(illustrate): diagram engine overhaul
- Files:
  - `workflows/shared/diagram_utils/registry.py`
  - `workflows/output/illustrate/nodes/generate_additional.py`
