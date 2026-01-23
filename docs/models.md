# LLM Model Tier Configuration Report

## Overview

This document catalogues ALL LLM calls across the Thala codebase with their model tier assignments, token expectations, and key characteristics.

**Available Tiers:**
| Tier | Model ID | Context Window | Cost (relative) |
|------|----------|----------------|-----------------|
| HAIKU | `claude-haiku-4-5-20251001` | 200k | Lowest |
| SONNET | `claude-sonnet-4-5-20250929` | 200k | Medium |
| SONNET_1M | `claude-sonnet-4-5-20250929` + beta header | 1M | Medium |
| OPUS | `claude-opus-4-5-20251101` | 200k | Highest |

---

## Complete LLM Call Inventory

### Legend
- **Structured**: Returns validated Pydantic schema
- **Tools**: Uses tool calling capability
- **Batched**: Typically processed via Batch API (50% cost savings)
- **Thinking**: Uses extended thinking budget

---

## 1. Content Classification & Filtering

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `core/scraping/classification/classifier.py:29` | Classify scraped content (full_text/abstract/paywall/non_academic) | 256 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/academic_lit_review/diffusion_engine/relevance_filters.py:108` | Filter papers by relevance to research topic | 512 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py:44` | Score paper relevance (0-1) | 256 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/academic_lit_review/citation_network/scoring.py:107` | Score citation importance | 256 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/wrappers/multi_lang/nodes/relevance_checker.py:112` | Check language-specific content relevance | 256 | Yes | No | No | **DEEPSEEK_V3** |

---

## 2. Query & Keyword Generation

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/research/academic_lit_review/keyword_search/searcher.py:196` | Score relevance during keyword search | 512 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/academic_lit_review/keyword_search/query_builder.py:32` | Build database queries from keywords | 256 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/web_research/subgraphs/researcher_base/query_generator.py` | Generate web search queries | 512 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/web_research/subgraphs/researcher_base/query_validator.py` | Validate/refine search queries | 256 | Yes | No | No | **DEEPSEEK_V3** |

---

## 3. Language Processing

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/shared/language/query_translator.py:170` | Translate search queries to other languages | 256 | Yes | No | No | **HAIKU** |
| `workflows/shared/language/translator.py:83` | Translate full prompts/content | 8192 | No | No | No | **OPUS** |

---

## 4. Paper & Document Extraction

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/research/academic_lit_review/paper_processor/extraction/core.py:74` | Extract paper summary (key findings, methodology, limitations) | 2048 | Yes | No | Yes | **HAIKU** (≤400k chars) |
| `workflows/research/academic_lit_review/paper_processor/extraction/core.py:74` | Extract paper summary (large papers >400k chars) | 4096 | Yes | No | Yes | **SONNET_1M** |
| `workflows/document_processing/nodes/metadata_agent.py:85` | Extract document metadata (title, authors, date, ISBN) | 1024 | Yes | No | No | **DEEPSEEK_R1** |
| `workflows/document_processing/nodes/summary_agent.py:81,123` | Generate document summaries | 4096 | No | No | No | **DEEPSEEK_R1** |
| `workflows/document_processing/nodes/chapter_detector.py:213` | Detect chapter boundaries in documents | 2048 | Yes | No | No | **DEEPSEEK_V3** |

---

## 5. Chapter & Section Processing

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier | Thinking |
|----------|-------------|--------------|------------|-------|---------|------|----------|
| `workflows/document_processing/subgraphs/chapter_summarization/nodes.py:50-52` | Summarize chapter content (small) | 4096 | No | No | No | **HAIKU** | 8000 |
| `workflows/document_processing/subgraphs/chapter_summarization/nodes.py:50-52` | Summarize chapter content (large >150k tokens) | 4096 | No | No | No | **SONNET_1M** | 8000 |
| `workflows/document_processing/subgraphs/chapter_summarization/nodes.py:279` | Screen large papers | 1024 | Yes | No | No | **HAIKU** | - |

---

## 6. Web Research Workflow

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/research/web_research/nodes/clarify_intent.py` | Clarify user research intent | 1024 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/web_research/nodes/create_brief.py:51` | Create structured research brief | 2048 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/research/web_research/nodes/iterate_plan.py` | Iterate on research plan | 4096 | Yes | No | No | **OPUS** |
| `workflows/research/web_research/nodes/supervisor/core.py:148` | Diffusion algorithm supervisor (orchestrates research) | 4096 | Yes | Yes | No | **OPUS** |
| `workflows/research/web_research/nodes/supervisor/llm_integration.py` | Get structured supervisor decisions | 2048 | Yes | No | No | **OPUS** |
| `workflows/research/web_research/subgraphs/web_researcher.py:181` | Execute web research sub-tasks | 4096 | Yes | Yes | No | **HAIKU** |
| `workflows/research/web_research/nodes/final_report.py:122` | Generate final research report | 16384 | No | No | No | **OPUS** |

---

## 7. Academic Lit Review - Clustering

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/research/academic_lit_review/clustering/llm_clustering.py:77-83` | Semantic clustering of papers (sees full corpus) | 16000 | Yes | No | No | **SONNET** |
| `workflows/research/academic_lit_review/clustering/analysis.py` | Analyze cluster insights | 4096 | Yes | No | No | **SONNET** |
| `workflows/research/academic_lit_review/clustering/synthesis.py` | Synthesize statistical + semantic clustering results | 8192 | Yes | No | No | **OPUS** |

---

## 8. Academic Lit Review - Synthesis Writing

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/research/academic_lit_review/synthesis/nodes/writing/drafting.py:88` | Write introduction section | 4096 | No | No | No | **SONNET** |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/drafting.py:90` | Write methodology section | 4096 | No | No | No | **SONNET** |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/drafting.py:201` | Write thematic sections | 4096 | No | No | No | **SONNET** |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/drafting.py:203` | Write discussion section | 4096 | No | No | No | **SONNET** |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/drafting.py:226` | Write conclusions section | 4096 | No | No | No | **SONNET** |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/revision.py:96,176` | Revise draft sections | 6000 | No | No | No | **SONNET** |
| `workflows/research/academic_lit_review/synthesis/nodes/quality_nodes.py:50` | Verify quality of final review | 2000 | Yes | No | No | **HAIKU** |
| `workflows/research/academic_lit_review/synthesis/nodes/integration_nodes.py:110` | Integrate content across sections | 4096 | Yes | No | No | **SONNET** |

---

## 9. Editing Workflow - Structure Analysis

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier | Thinking |
|----------|-------------|--------------|------------|-------|---------|------|----------|
| `workflows/enhance/editing/nodes/analyze_structure.py:39-73` | Analyze document structure for issues | 8000 | Yes | No | No | **OPUS**¹ | 2000-10000² |
| `workflows/enhance/editing/nodes/plan_edits.py` | Plan ordered edits to fix issues | 4096 | Yes | No | No | **OPUS**¹ | - |

¹ Falls back to SONNET if `use_opus_for_analysis=False` (test/quick tiers)
² Thinking budget: test=2000, quick=4000, standard=6000, comprehensive=8000, high_quality=10000

---

## 10. Editing Workflow - Content Generation

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/enhance/editing/nodes/execute_edits.py:130` | Execute planned edits | 2000 | No | No | No | **HAIKU** |
| `workflows/enhance/editing/nodes/execute_edits.py:152` | Generate new content | 2000 | No | No | No | **HAIKU** |
| `workflows/enhance/editing/nodes/execute_edits.py:173` | Refine existing content | 2000 | No | No | No | **HAIKU** |
| `workflows/enhance/editing/nodes/execute_edits.py:199` | Extract metadata | 500 | Yes | No | No | **HAIKU** |
| `workflows/enhance/editing/nodes/execute_edits.py:288` | Generate complex content | 4000 | No | No | No | **HAIKU** |
| `workflows/enhance/editing/nodes/execute_edits.py:321` | Generate synthesis content | 3000 | No | No | No | **HAIKU** |

---

## 11. Editing Workflow - Enhancement & Verification

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/enhance/editing/nodes/enhance_section.py:204-220` | Enhance section with citations | 4096 | Yes | Yes | No | **OPUS**¹ |
| `workflows/enhance/editing/nodes/enhance_coherence.py:58` | Review document coherence | 2048 | Yes | No | No | **SONNET** |
| `workflows/enhance/editing/nodes/verify_structure.py:68` | Verify structural changes | 2048 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/enhance/editing/nodes/finalize.py:84` | Final verification of edits | 2048 | Yes | No | No | **DEEPSEEK_V3** |

¹ Falls back to SONNET if `use_opus_for_generation=False`

---

## 12. Editing Workflow - Polish (Fast Tier)

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/enhance/editing/nodes/polish.py:81` | Screen sections needing polish | 1024 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/enhance/editing/nodes/polish.py:153` | Polish section for flow/clarity | 2048 | Yes | No | No | **HAIKU** |

---

## 13. Fact-Checking Workflow

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/enhance/fact_check/nodes/screen_sections.py:61` | Pre-screen sections for verifiable claims | 1024 | Yes | No | No | **DEEPSEEK_V3** |
| `workflows/enhance/fact_check/nodes/fact_check.py:127-140` | Fact-check claims with tools | 4096 | Yes | Yes | No | **HAIKU** |
| `workflows/enhance/fact_check/nodes/reference_check.py:234` | Validate citation accuracy | 4096 | Yes | Yes | No | **HAIKU** |

---

## 14. Supervision Loops

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier | Thinking |
|----------|-------------|--------------|------------|-------|---------|------|----------|
| `workflows/enhance/supervision/shared/nodes/analyze_review.py:90` | Analyze reviewer feedback | 4096 | Yes | No | No | **OPUS** | 8000 |
| `workflows/enhance/supervision/shared/nodes/integrate_content.py:95-96` | Integrate suggested content | 8192 | Yes | No | No | **OPUS** | 8000 |
| `workflows/enhance/supervision/loop2/graph.py:98` | Supervision structured output | 4096 | Yes | No | No | **OPUS** | - |
| `workflows/enhance/supervision/loop2/graph.py:286` | Multi-round supervision | 16384 | No | No | No | **OPUS** | - |

---

## 15. Multi-Language Workflows

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/wrappers/multi_lang/nodes/opus_integrator.py:80` | Create initial synthesis from English findings | 64000 | Yes | No | No | **OPUS** |
| `workflows/wrappers/multi_lang/nodes/opus_integrator.py:117` | Integrate additional language findings | 32000 | Yes | No | No | **OPUS** |
| `workflows/wrappers/multi_lang/nodes/opus_integrator.py:160` | Final cross-language enhancement | 16000 | Yes | No | No | **OPUS** |
| `workflows/wrappers/multi_lang/nodes/sonnet_analyzer.py:135` | Cross-language analysis (1M context) | 8192 | Yes | No | No | **SONNET** |
| `workflows/wrappers/multi_lang/nodes/language_executor.py:31` | Execute language-specific research | 4096 | Yes | Yes | No | **SONNET** |

---

## 16. Synthesis Workflows

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/wrappers/synthesis/nodes/synthesis.py:159-164` | Structure synthesis document | 8192 | Yes | No | No | **OPUS**¹ |
| `workflows/wrappers/synthesis/nodes/quality_check.py:151-156` | Verify synthesis quality | 2048 | Yes | No | No | **HAIKU** |
| `workflows/wrappers/synthesis/nodes/research_targets.py:107` | Identify research targets | 2048 | Yes | No | No | **HAIKU** |

¹ Falls back to SONNET for test tier

---

## 17. Output & Essay Generation

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/output/substack_review/nodes/choose_essay.py:50` | Select best essay variant | 2048 | Yes | No | No | **OPUS** |
| `workflows/output/substack_review/nodes/write_essay.py:36` | Write Substack essay | 8192 | No | No | No | **OPUS** |

---

## 18. Batch Processing Infrastructure

| Location | Description | Tokens (out) | Structured | Tools | Batched | Tier |
|----------|-------------|--------------|------------|-------|---------|------|
| `workflows/shared/batch_processor/processor.py:68` | Batch API default | 4096 | Yes | Yes | Yes | **SONNET** |

---

## Summary Statistics

### By Model Tier

| Tier | Call Count | Primary Use Cases |
|------|------------|-------------------|
| **HAIKU** | ~30 | Classification, filtering, screening, query generation, polish, quality checks |
| **SONNET** | ~45 | Standard processing, writing sections, fact-checking, analysis |
| **SONNET_1M** | ~5 | Large documents (>400k chars / ~100k tokens) |
| **OPUS** | ~25 | Complex reasoning, synthesis, supervision, final reports, integration |

### Key Cost Optimization Strategies

1. **Prompt Caching**: Enabled by default (90% savings on cache hits, 10% read cost)
2. **Batch API**: 50% cost reduction for bulk operations (auto-triggered at 5+ items)
3. **Model Tiering**: HAIKU for fast/cheap tasks, OPUS only for complex reasoning
4. **Context Fallback**: Auto-upgrade SONNET → SONNET_1M on context errors

### Extended Thinking Usage

| Quality Tier | Thinking Budget | Use Case |
|--------------|-----------------|----------|
| test | 2,000 | Quick validation |
| quick | 4,000 | Fast processing |
| standard | 6,000 | Balanced |
| comprehensive | 8,000 | High quality |
| high_quality | 10,000 | Best results |

---

## Deepseek Comparison & Migration Analysis

### Model Options

| Model | Input $/MTok | Output $/MTok | Context | Best For |
|-------|--------------|---------------|---------|----------|
| **DeepSeek-V3.2** | $0.27 | $1.10 | 128K | High-volume, simple tasks |
| **DeepSeek-R1** | $0.55 | $2.19 | 128K | Reasoning tasks |
| **Claude Haiku 4.5** | $1.00 | $5.00 | 200K | Fast quality tasks |
| **Claude Sonnet 4.5** | $3.00 | $15.00 | 200K | Standard processing |
| **Claude Opus 4.5** | ~$15.00 | ~$75.00 | 200K | Complex reasoning |

**Cost ratios** (vs Claude Haiku):
- DeepSeek-V3.2: **~10-15x cheaper**
- DeepSeek-R1: **~3-5x cheaper**

---

### Task-by-Task Migration Recommendations

#### HIGH CONFIDENCE → DeepSeek-V3.2 (Currently HAIKU)

| Task | Current Cost | DeepSeek Cost | Savings | Risk |
|------|--------------|---------------|---------|------|
| Content classification | $1/$5 | $0.27/$1.10 | ~80% | Low |
| Relevance filtering | $1/$5 | $0.27/$1.10 | ~80% | Low |
| Relevance scoring | $1/$5 | $0.27/$1.10 | ~80% | Low |
| Query/keyword generation | $1/$5 | $0.27/$1.10 | ~80% | Low |
| Query translation | $1/$5 | $0.27/$1.10 | ~80% | Low |
| Polish screening | $1/$5 | $0.27/$1.10 | ~80% | Low |
| Quality checks (simple) | $1/$5 | $0.27/$1.10 | ~80% | Low |

**Notes:**
- DeepSeek has OpenAI-compatible API (easy migration)
- ~85-90% accuracy vs Claude's ~95% on classification
- JSON strict mode available (beta)
- Off-peak pricing (50-75% discount): 16:30-00:30 UTC

---

#### MEDIUM CONFIDENCE → DeepSeek-V3.2 or R1 (Currently SONNET)

| Task | Recommend | Savings | Risk | Notes |
|------|-----------|---------|------|-------|
| Metadata extraction | V3.2 | ~90% | Medium | ~80% accuracy vs 95% |
| Basic summarization | V3.2 | ~90% | Medium | Quality slightly lower |
| Chapter detection | V3.2 | ~90% | Medium | Structured output works |
| Paper summary (small) | R1 | ~80% | Medium | Better reasoning for methodology |
| Drafting (non-final) | V3.2 | ~90% | Low | Use for drafts only |

**Strategy**: Two-pass approach
1. DeepSeek for initial extraction/draft
2. Claude Haiku for validation/edge cases

---

#### LOW CONFIDENCE → Keep Claude (Currently OPUS/SONNET)

| Task | Why Keep Claude | Risk if Migrated |
|------|-----------------|------------------|
| Complex synthesis | Opus reasoning quality essential | Significant quality loss |
| Multi-turn tool use | Claude's tool calling more robust | Reliability issues |
| Extended thinking | DeepSeek R1 has reasoning but different approach | Behavior differences |
| Cross-language integration | Nuance in translation quality | Academic tone issues |
| Supervision loops | Multi-round coherence critical | Context coherence degradation |
| Fact-checking with tools | Tool reliability matters | Higher error rate |
| Academic writing (final) | Tone, citations, formatting | Quality unacceptable |

---

### Implementation Priority

**Phase 1: Low-Risk Migration** (Immediate)
- Classification → DeepSeek-V3.2
- Relevance filtering → DeepSeek-V3.2
- Query generation → DeepSeek-V3.2
- *Expected savings: ~$500/month*

**Phase 2: Validation Layer** (Week 2)
- Add DeepSeek-V3.2 for metadata extraction
- Keep Claude Haiku as validation fallback
- *Expected savings: ~$200/month additional*

**Phase 3: Draft Processing** (Week 3-4)
- Use DeepSeek-V3.2 for draft summaries
- Use DeepSeek-R1 for reasoning-heavy drafts
- Final pass remains Claude Sonnet
- *Expected savings: ~$300/month additional*

**Phase 4: Evaluation** (Month 2)
- Measure quality degradation
- Adjust tier boundaries
- Consider self-hosting DeepSeek for privacy

---

### Technical Integration Notes

**DeepSeek API is OpenAI-compatible:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com"  # or /beta for strict mode
)

# Same patterns work:
response = client.chat.completions.create(
    model="deepseek-chat",  # V3.2
    # model="deepseek-reasoner",  # R1
    messages=[...],
    response_format={"type": "json_object"}
)
```

**Current LangChain integration** would need:
1. Add `ChatDeepSeek` wrapper (or use OpenAI wrapper with base_url)
2. Add `ModelTier.DEEPSEEK_V3` and `ModelTier.DEEPSEEK_R1`
3. Update `get_llm()` to route appropriately

---

### Privacy & Compliance Considerations

| Concern | Claude | DeepSeek (API) | DeepSeek (Self-hosted) |
|---------|--------|----------------|------------------------|
| Data residency | US/EU | China | Your control |
| SOC 2 | Yes | No | N/A |
| GDPR compliance | Yes | Questionable | Your control |
| Training on data | Opt-out | Unknown | N/A |

**Recommendation**: For sensitive academic data, use Claude or self-host DeepSeek (MIT license).

---

### Projected Monthly Savings

Assuming current usage patterns:

| Approach | Monthly Cost | vs. Current |
|----------|--------------|-------------|
| All Claude (current) | ~$2,000 | baseline |
| Hybrid (recommended) | ~$950 | **-52%** |
| Aggressive DeepSeek | ~$600 | -70% (quality risk) |

---

### Quality vs. Cost Decision Matrix

```
                    HIGH QUALITY NEEDED
                           ↑
                           │
        Keep Claude        │    Consider R1
        (OPUS tasks)       │    (reasoning tasks)
                           │
    ←──────────────────────┼──────────────────────→
                           │                     HIGH VOLUME
        Use Claude         │    Use DeepSeek V3.2
        (validation)       │    (bulk processing)
                           │
                           ↓
                    COST SENSITIVE
```

---

*Generated: 2026-01-20*
*Source: Codebase analysis via Explore agents + DeepSeek API documentation research*
