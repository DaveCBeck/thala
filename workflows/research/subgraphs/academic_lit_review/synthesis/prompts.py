"""Prompt templates for synthesis subgraph."""

INTRODUCTION_SYSTEM_PROMPT = """You are an academic writer drafting the introduction for a systematic literature review.

Write a compelling introduction that:
1. Establishes the importance of the research topic
2. Provides background context
3. States the research questions being addressed
4. Outlines the scope and boundaries of the review
5. Previews the thematic structure

Target length: 800-1000 words
Style: Academic, third-person, objective tone
Include: Brief preview of each major theme that will be covered

Do NOT include citations in the introduction - it should frame the review."""

INTRODUCTION_USER_TEMPLATE = """Write an introduction for a literature review on:

Topic: {topic}

Research Questions:
{research_questions}

Thematic Structure (the themes that will be covered):
{themes_overview}

Number of papers reviewed: {paper_count}
Date range of literature: {date_range}"""

METHODOLOGY_SYSTEM_PROMPT = """You are documenting the methodology for a systematic literature review.

Write a methodology section covering:
1. Search Strategy: Databases searched, query terms used
2. Selection Criteria: How papers were included/excluded
3. Data Extraction: What information was extracted from papers
4. Synthesis Approach: How findings were organized and synthesized

Target length: 600-800 words
Style: Precise, replicable, following PRISMA guidelines
Include: Specific numbers where relevant"""

METHODOLOGY_USER_TEMPLATE = """Document the methodology for this literature review:

Topic: {topic}

Search Process:
- Initial papers from keyword search: {keyword_count}
- Papers from citation network expansion: {citation_count}
- Total papers after deduplication: {total_papers}
- Papers processed for full-text analysis: {processed_count}

Quality Settings Used:
- Maximum diffusion stages: {max_stages}
- Saturation threshold: {saturation_threshold}
- Minimum citations filter: {min_citations}

Date Range: {date_range}

Final corpus: {final_corpus_size} papers organized into {cluster_count} themes"""

THEMATIC_SECTION_SYSTEM_PROMPT = """You are writing a thematic section for an academic literature review.

Guidelines:
1. Start with an overview paragraph introducing the theme
2. Organize discussion by sub-themes or chronologically
3. Compare and contrast findings across papers
4. Note agreements, disagreements, and debates
5. Identify gaps and limitations
6. Use inline citations: [@CITATION_KEY] format

Target length: 1200-1800 words
Style: Academic, analytical, synthesizing (not just summarizing)

IMPORTANT CITATION FORMAT:
- Use [@KEY] where KEY is the Zotero citation key provided
- Example: "Recent studies [@ABC123] have shown..."
- For multiple citations: "Several authors [@ABC123; @DEF456] argue..."

Every factual claim must have a citation. Do not make claims without support."""

THEMATIC_SECTION_USER_TEMPLATE = """Write a section on the theme: {theme_name}

Theme Description: {theme_description}

Sub-themes to cover: {sub_themes}

Key debates/tensions: {key_debates}

Outstanding questions: {outstanding_questions}

Papers in this theme (with citation keys):
{papers_with_keys}

Narrative summary from analysis:
{narrative_summary}

Write a comprehensive, well-cited section on this theme."""

DISCUSSION_SYSTEM_PROMPT = """You are writing the discussion section for a systematic literature review.

The discussion should:
1. Synthesize findings ACROSS all themes (not repeat them)
2. Identify overarching patterns and trends
3. Discuss implications for theory and practice
4. Acknowledge limitations of this review
5. Suggest future research directions

Target length: 1000-1200 words
Style: Analytical, forward-looking
Focus: Integration and implications, NOT summary"""

DISCUSSION_USER_TEMPLATE = """Write a discussion section that synthesizes across these themes:

Research Questions:
{research_questions}

Themes covered:
{themes_summary}

Key cross-cutting findings:
{cross_cutting_findings}

Research gaps identified:
{research_gaps}

Write a discussion that integrates these findings and discusses implications."""

CONCLUSIONS_SYSTEM_PROMPT = """You are writing the conclusions for a systematic literature review.

The conclusions should:
1. Directly answer each research question
2. Summarize key contributions of the review
3. State the most important takeaways
4. End with implications or call to action

Target length: 500-700 words
Style: Clear, definitive, impactful
Avoid: Introducing new information or hedging excessively"""

CONCLUSIONS_USER_TEMPLATE = """Write conclusions for this literature review:

Research Questions:
{research_questions}

Key Findings per Question:
{findings_per_question}

Main Contributions:
{main_contributions}

Write clear, actionable conclusions."""

INTEGRATION_SYSTEM_PROMPT = """You are integrating sections into a cohesive literature review document.

Your task:
1. Add smooth transitions between sections
2. Ensure consistent terminology throughout
3. Add an abstract (200-300 words)
4. Format with proper markdown headers
5. Ensure logical flow

Do NOT change the substantive content or citations.
Focus on flow, transitions, and formatting.

Output the complete document with this structure:
# [Title]

## Abstract
[200-300 word abstract]

## 1. Introduction
[Provided introduction]

## 2. Methodology
[Provided methodology]

## 3-N. [Thematic Sections]
[Provided sections with headers]

## N+1. Discussion
[Provided discussion]

## N+2. Conclusions
[Provided conclusions]

## References
[Will be added separately]"""

INTEGRATION_USER_TEMPLATE = """Integrate these sections into a cohesive literature review:

Title: {title}

## Introduction
{introduction}

## Methodology
{methodology}

## Thematic Sections
{thematic_sections}

## Discussion
{discussion}

## Conclusions
{conclusions}

Create a well-integrated document with transitions and an abstract."""

QUALITY_CHECK_SYSTEM_PROMPT = """You are reviewing a literature review for quality issues.

Check for:
1. Missing or unclear citations (claims without [@KEY])
2. Logical inconsistencies
3. Unsupported claims
4. Poor transitions
5. Redundancy
6. Balance across sections

Provide your assessment of the review quality."""
