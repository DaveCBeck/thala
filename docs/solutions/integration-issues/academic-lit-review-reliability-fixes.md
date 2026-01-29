---
module: workflows/research/subgraphs/academic_lit_review
date: 2025-12-30
problem_type: api_integration_issue
component: clustering, diffusion_engine, paper_processor, openalex
symptoms:
  - "JSON parsing failures in clustering when LLM returns unexpected formatting"
  - "Co-cited papers missing from corpus due to incomplete DOI lookups"
  - "Rate-limiting errors from excessive concurrent requests to retrieve-academic"
  - "Workflow failures when papers discovered via citation analysis aren't retrievable"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags: [academic-lit-review, llm-reliability, pydantic, structured-output, doi-lookup, rate-limiting, unified-pipeline, openalex]
---

# Academic Literature Review Reliability Fixes

## Problem

The academic literature review workflow experienced cascading reliability failures across multiple components:

1. **JSON parsing failures**: LLM clustering output used fragile regex-based JSON extraction, failing on unexpected formatting (missing code fences, extra whitespace, partial responses)
2. **Missing co-cited papers**: Citation diffusion discovered papers via co-citation analysis, but couldn't retrieve their metadata if they weren't in the original search results
3. **Rate-limiting errors**: Separate acquisition and processing phases with independent concurrency caused excessive parallel requests, overwhelming external APIs
4. **Pipeline inefficiency**: GPU sat idle during acquisition while separate phases created bottlenecks

### Symptoms

```
# JSON parsing failure in clustering
ERROR: Failed to parse LLM clustering response: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
ERROR: LLM clustering failed, falling back to BERTopic-only

# Missing co-cited papers
WARNING: Co-cited paper 10.1038/nature12373 not found in candidates
WARNING: 47 co-cited papers could not be added to corpus (missing metadata)

# Rate limiting
ERROR: 429 Too Many Requests from retrieve-academic
ERROR: Acquisition task failed: httpx.HTTPStatusError: 429 for url ...

# Pipeline inefficiency (observed in LangSmith traces)
INFO: Acquisition complete (120s), starting processing...
# 120s of GPU idle time while waiting for all acquisitions
```

## Root Cause

### Issue 1: Fragile JSON Extraction

Clustering used manual regex parsing that failed on valid but non-standard LLM output:

```python
# BEFORE: Fragile regex extraction
content = response.content
if content.startswith("```"):
    lines = content.split("\n")
    content = "\n".join(lines[1:-1])  # Fails if no ending ```

result = json.loads(content)  # Fails on partial JSON, extra whitespace
themes = [
    LLMTheme(
        name=theme_data.get("name", "Unnamed"),  # No validation
        paper_dois=theme_data.get("paper_dois", []),  # Could be None
    )
    for theme_data in result.get("themes", [])
]
```

### Issue 2: No DOI Lookup for Citation-Discovered Papers

Citation diffusion tracked DOIs but couldn't fetch metadata for papers not in original search:

```python
# BEFORE: Only used candidates from search
for doi in cocitation_included:
    if doi not in candidate_lookup:
        logger.warning(f"Co-cited paper {doi} not found in candidates")
        # Paper lost from corpus!
```

### Issue 3: Independent Concurrency Without Rate Limiting

Acquisition and processing had separate semaphores, causing burst requests:

```python
# BEFORE: Two separate phases with independent concurrency
acquired, failed = await acquire_all_papers(papers, max_concurrent=5)
results, failed = await process_papers_batch(acquired, max_concurrent=3)
# 5 + 3 = 8 concurrent requests bursting at start of each phase
```

## Solution

### Step 1: Replace JSON Parsing with Pydantic Structured Output

Define strict Pydantic models for all clustering outputs:

```python
# workflows/research/subgraphs/academic_lit_review/clustering.py

from pydantic import BaseModel, Field


class LLMThemeOutput(BaseModel):
    """Pydantic model for a single theme from LLM clustering."""

    name: str = Field(description="Clear, descriptive theme name")
    description: str = Field(description="2-3 sentence description")
    paper_dois: list[str] = Field(description="DOIs belonging to this theme")
    sub_themes: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class LLMTopicSchemaOutput(BaseModel):
    """Pydantic model for LLM semantic clustering output."""

    themes: list[LLMThemeOutput] = Field(description="Identified themes")
    reasoning: str = Field(description="Clustering rationale")


class ClusterAnalysisOutput(BaseModel):
    """Pydantic model for deep analysis of a single cluster."""

    narrative_summary: str = Field(description="2-3 paragraph summary")
    timeline: list[str] = Field(default_factory=list)
    key_debates: list[str] = Field(default_factory=list)
    methodologies: list[str] = Field(default_factory=list)
    outstanding_questions: list[str] = Field(default_factory=list)


class OpusSynthesisOutput(BaseModel):
    """Pydantic model for Opus cluster synthesis output."""

    reasoning: str = Field(description="Synthesis decisions")
    final_clusters: list[ThematicClusterOutput] = Field(description="Final clusters")
```

Use `with_structured_output()` instead of JSON parsing:

```python
# AFTER: Pydantic structured output
async def run_llm_clustering_node(state: ClusteringState) -> dict[str, Any]:
    llm = get_llm(tier=ModelTier.SONNET, max_tokens=16000)

    # Use structured output - automatic validation
    structured_llm = llm.with_structured_output(LLMTopicSchemaOutput)
    messages = [
        {"role": "system", "content": LLM_CLUSTERING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result: LLMTopicSchemaOutput = await structured_llm.ainvoke(messages)

    # Convert Pydantic to TypedDict for state compatibility
    themes: list[LLMTheme] = []
    for theme in result.themes:  # Type-safe attribute access
        themes.append(
            LLMTheme(
                name=theme.name,  # Guaranteed non-None
                description=theme.description,
                paper_dois=theme.paper_dois,  # Guaranteed list
                sub_themes=theme.sub_themes,
                relationships=theme.relationships,
            )
        )

    return {"llm_topic_schema": LLMTopicSchema(themes=themes, reasoning=result.reasoning)}
```

### Step 2: Add Batch DOI Lookup Functions

Add functions to fetch paper metadata by DOI:

```python
# langchain_tools/openalex.py

async def get_work_by_doi(doi: str) -> Optional[OpenAlexWork]:
    """Fetch a work's full metadata by DOI.

    Useful for papers discovered via citation analysis but not in search results.
    """
    client = _get_openalex()
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    try:
        work_url = f"/works/doi:{doi_clean}"
        response = await client.get(work_url)
        response.raise_for_status()
        work_data = response.json()

        parsed = _parse_work(work_data)
        logger.debug(f"Fetched work by DOI {doi_clean}: {parsed.title[:50]}...")
        return parsed

    except Exception as e:
        logger.debug(f"get_work_by_doi failed for {doi_clean}: {e}")
        return None


async def get_works_by_dois(dois: list[str]) -> list[OpenAlexWork]:
    """Fetch multiple works by DOIs in a single batch request.

    Uses pipe-delimited filter for efficiency: doi:id1|id2|id3
    """
    if not dois:
        return []

    client = _get_openalex()
    dois_clean = [
        d.replace("https://doi.org/", "").replace("http://doi.org/", "")
        for d in dois
    ]

    try:
        # Pipe-delimited filter - one request for N DOIs
        filter_str = "|".join(f"https://doi.org/{d}" for d in dois_clean)
        params = {
            "filter": f"doi:{filter_str}",
            "per_page": min(len(dois_clean), 50),
        }

        response = await client.get("/works", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for work in data.get("results", []):
            try:
                parsed = _parse_work(work)
                results.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse work in batch: {e}")

        logger.debug(f"Batch fetched {len(results)}/{len(dois)} works by DOI")
        return results

    except Exception as e:
        logger.error(f"get_works_by_dois failed: {e}")
        return []
```

### Step 3: Add DOI Fallback for Co-cited Papers

Use batch DOI lookup in diffusion engine:

```python
# workflows/research/subgraphs/academic_lit_review/diffusion_engine.py

async def update_corpus_and_graph(state: DiffusionEngineState) -> dict[str, Any]:
    cocitation_included = state.get("cocitation_included", [])
    candidate_lookup = {p.get("doi"): p for p in state.get("current_stage_candidates", [])}
    all_candidates_lookup = {p.get("doi"): p for p in state.get("all_stage_candidates", [])}

    # Find co-cited papers missing from candidates
    missing_cocited_dois = [
        doi for doi in cocitation_included
        if doi not in candidate_lookup and doi not in all_candidates_lookup
    ]

    # DOI fallback: fetch missing papers from OpenAlex
    fallback_papers = {}
    if missing_cocited_dois:
        logger.info(f"Fetching {len(missing_cocited_dois)} co-cited papers by DOI fallback")
        try:
            fetched_works = await get_works_by_dois(missing_cocited_dois)
            for work in fetched_works:
                if work.doi:
                    doi_clean = work.doi.replace("https://doi.org/", "")
                    paper = convert_to_paper_metadata(
                        work.model_dump(),
                        discovery_stage=diffusion["current_stage"],
                        discovery_method="cocitation_fallback",
                    )
                    if paper:
                        fallback_papers[doi_clean] = paper
                        logger.debug(f"Recovered co-cited paper: {work.title[:50]}...")

            # Log still-missing papers
            still_missing = set(missing_cocited_dois) - set(fallback_papers.keys())
            for doi in still_missing:
                logger.warning(f"Co-cited paper {doi} not found in OpenAlex")

        except Exception as e:
            logger.warning(f"Failed to fetch co-cited papers by DOI: {e}")

    # Add relevant papers to corpus (includes fallback)
    new_corpus_papers = {}
    for doi in all_relevant_dois:
        paper = (
            candidate_lookup.get(doi)
            or all_candidates_lookup.get(doi)
            or fallback_papers.get(doi)  # NEW: DOI fallback
        )
        if paper:
            new_corpus_papers[doi] = paper

    return {"paper_corpus": new_corpus_papers}
```

### Step 4: Unify Acquisition and Processing Pipeline

Combine separate phases into a single pipeline with natural rate limiting:

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor.py

MAX_PAPER_PIPELINE_CONCURRENT = 2  # Low concurrency prevents API overwhelm
ACQUISITION_DELAY = 2.0  # Delay between requests


async def acquire_and_process_single_paper(
    paper: PaperMetadata,
    client: RetrieveAcademicClient,
    output_dir: Path,
    paper_index: int,
    total_papers: int,
) -> dict[str, Any]:
    """Acquire and process a single paper as one unit.

    Combining acquisition + processing naturally rate-limits:
    processing takes time, giving external APIs a break.
    """
    doi = paper.get("doi")
    title = paper.get("title", "Unknown")[:50]

    logger.info(f"[{paper_index}/{total_papers}] Processing: {title}...")

    result = {
        "doi": doi,
        "acquired": False,
        "processing_success": False,
    }

    # Step 1: Acquire full text
    local_path = await acquire_full_text(paper, client, output_dir)
    if not local_path:
        logger.warning(f"[{paper_index}/{total_papers}] Acquisition failed for {doi}")
        return result

    result["acquired"] = True
    result["local_path"] = local_path

    # Step 2: Process (takes time, naturally rate-limits)
    processing_result = await process_single_document(doi, local_path, paper)
    result["processing_result"] = processing_result
    result["processing_success"] = processing_result.get("success", False)

    return result


async def run_paper_pipeline(
    papers: list[PaperMetadata],
    max_concurrent: int = MAX_PAPER_PIPELINE_CONCURRENT,
) -> tuple[dict, dict, list, list]:
    """Unified acquire→process pipeline with controlled concurrency."""
    async with RetrieveAcademicClient() as client:
        if not await client.health_check():
            return {}, {}, [p.get("doi") for p in papers], []

        output_dir = Path("/tmp/thala_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrent)
        total_papers = len(papers)

        async def process_with_limit(paper: PaperMetadata, index: int) -> dict:
            async with semaphore:
                # Rate limiting delay
                if index > 0:
                    await asyncio.sleep(ACQUISITION_DELAY)
                return await acquire_and_process_single_paper(
                    paper, client, output_dir, index + 1, total_papers
                )

        tasks = [process_with_limit(paper, i) for i, paper in enumerate(papers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results...
        acquired = {}
        processing_results = {}
        acquisition_failed = []
        processing_failed = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Pipeline task failed: {result}")
                continue

            doi = result.get("doi")
            if result.get("acquired"):
                acquired[doi] = result.get("local_path")
                if result.get("processing_success"):
                    processing_results[doi] = result.get("processing_result")
                else:
                    processing_failed.append(doi)
            else:
                acquisition_failed.append(doi)

        return acquired, processing_results, acquisition_failed, processing_failed
```

Update the graph to use unified pipeline:

```python
# workflows/research/subgraphs/academic_lit_review/paper_processor.py

def create_paper_processing_subgraph() -> StateGraph:
    """Create paper processing subgraph with unified pipeline.

    Flow: START -> acquire_and_process -> extract_summaries -> END

    The unified pipeline naturally rate-limits because processing takes time.
    """
    builder = StateGraph(PaperProcessingState)

    # Combined node replaces separate acquire + process
    builder.add_node("acquire_and_process", acquire_and_process_papers_node)
    builder.add_node("extract_summaries", extract_summaries_node)

    builder.add_edge(START, "acquire_and_process")
    builder.add_edge("acquire_and_process", "extract_summaries")
    builder.add_edge("extract_summaries", END)

    return builder.compile()
```

## Prevention

### Structured Output Guidelines

1. **Always use Pydantic models** for complex LLM outputs
2. **Use `with_structured_output()`** instead of JSON regex extraction
3. **Add `default_factory=list`** for optional list fields to avoid None
4. **Convert Pydantic to TypedDict** at state boundaries for LangGraph compatibility

### DOI Lookup Guidelines

1. **Batch DOI lookups** when possible using pipe-delimited filter
2. **Add DOI fallback** for citation-discovered papers
3. **Track discovery method** (`cocitation_fallback`) for audit trail
4. **Log missing papers** for debugging incomplete corpora

### Rate Limiting Guidelines

1. **Use unified pipelines** that combine acquisition + processing
2. **Low concurrent limits** (2-3) for external APIs
3. **Add delays between requests** (`ACQUISITION_DELAY = 2.0`)
4. **Natural rate limiting**: long processing tasks give APIs a break

## Files Modified

- `langchain_tools/openalex.py`: Added `get_work_by_doi()`, `get_works_by_dois()`
- `workflows/research/subgraphs/academic_lit_review/clustering.py`: Pydantic models, structured output
- `workflows/research/subgraphs/academic_lit_review/diffusion_engine.py`: DOI fallback for co-cited papers
- `workflows/research/subgraphs/academic_lit_review/paper_processor.py`: Unified pipeline

## Related Patterns

- [Citation Network Academic Review Workflow](../../patterns/langgraph/citation-network-academic-review-workflow.md) - Overall workflow architecture
- [Concurrent Scraping with TTL Cache](../../patterns/async-python/concurrent-scraping-with-ttl-cache.md) - Async patterns for parallel acquisition

## Related Solutions

- [Query Generation and Supervisor Extraction Fixes](../llm-output/query-generation-supervisor-extraction-fixes.md) - Structured output patterns
- [Workflow Completeness Tracking and Reliability Fixes](../langgraph-issues/workflow-completeness-tracking-reliability-fixes.md) - Retry and transient error handling

## References

- [Pydantic with_structured_output](https://python.langchain.com/docs/how_to/structured_output/)
- [OpenAlex Filter API](https://docs.openalex.org/api-entities/works/filter-works)
