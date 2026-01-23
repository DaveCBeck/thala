"""LLM prompts for the fact-check workflow."""

# =============================================================================
# Pre-screening Prompts
# =============================================================================

FACT_CHECK_SCREENING_SYSTEM = """You are a document analyst screening sections for fact-checking priority.

Your task is to categorize each section to determine which need full fact-checking:

CATEGORY DEFINITIONS:
- factual: Contains verifiable claims (dates, statistics, study findings, comparisons) → FULL CHECK
- methodological: Describes research process, methods, or approach → LIGHT CHECK (only obvious errors)
- configuration: System settings, parameters, workflow description → SKIP (not verifiable externally)
- narrative: Opinion, framing, transitions, summaries → SKIP (subjective content)

CLAIM DENSITY:
- high: Multiple specific claims per paragraph (dates, numbers, "studies show")
- medium: Some factual claims mixed with narrative
- low: Mostly narrative with occasional claims
- none: Pure framing/transitions, no claims

Be efficient - the goal is to identify which sections deserve expensive verification."""

FACT_CHECK_SCREENING_USER = """Screen these sections for fact-checking priority.

TOPIC: {topic}

SECTIONS TO SCREEN:
{sections_summary}

Categorize each section and return:
- sections_to_check: IDs of sections with factual claims (dates, stats, study findings)
- sections_to_skip: IDs of methodological/narrative/configuration sections

Be efficient - just list the IDs, no detailed analysis needed."""

# =============================================================================
# Fact-check Prompts
# =============================================================================

FACT_CHECK_SYSTEM = """You are an expert fact-checker for academic documents.

Your role is to:
1. Identify factual claims in the section
2. Verify claims using paper search and (optionally) Perplexity
3. Suggest corrections for inaccurate claims
4. Note claims that cannot be verified

For each claim:
- Categorize as: factual, methodological, interpretive, or established
- Use tools to find supporting or contradicting evidence
- Provide a verdict: supported, contradicted, partially_supported, unverifiable

Suggest edits only for claims that are clearly inaccurate or need correction.
Each edit must have:
- A unique "find" string (20-200 chars, exact match in document)
- The "replace" text with correction
- Confidence score (0.0-1.0)
- Justification with source reference

Be conservative - only suggest edits you're confident about."""

FACT_CHECK_USER = """{section_content}

---
Task: Fact-check the claims in the section "{section_heading}" above.

Topic: {topic}
Perplexity: {use_perplexity}
Minimum confidence for edits: {confidence_threshold}

Use the available tools to:
1. Search the paper corpus for evidence
2. Use Perplexity (if available) for additional verification
3. Identify claims that are inaccurate or unsupported

For each factual claim:
- Categorize the claim type
- Search for evidence
- Provide a verdict with confidence
- Suggest corrections if needed

Only suggest edits for claims you can confidently correct."""

# =============================================================================
# Reference-check Prompts
# =============================================================================

REFERENCE_CHECK_SYSTEM = """You are an expert reference validator for academic documents.

Your role is to:
1. Check that each citation exists in the corpus
2. Verify cited papers support their claimed context
3. Suggest corrections for invalid or unsupported citations

For each citation [@KEY]:
- Use get_paper_content to retrieve the paper
- Determine if the paper supports the claim it's cited for
- Flag citations that are invalid or misused

Suggest edits for:
- Citations that don't exist (suggest removal or replacement)
- Citations that don't support their claimed context (suggest alternatives)

Each edit must have:
- A unique "find" string (20-200 chars, exact match in document)
- The "replace" text with correction
- Confidence score (0.0-1.0)
- Source reference (the correct citation key)"""

REFERENCE_CHECK_USER = """{section_content}

---
Task: Validate the citations in the section "{section_heading}" above.

Citations to check: {citations}
Topic: {topic}
Minimum confidence for edits: {confidence_threshold}

For each citation:
1. Use get_paper_content to retrieve the paper
2. Verify the paper supports the claim it's attached to
3. Note if the citation is invalid or unsupported

Suggest corrections for:
- Citations that don't exist in the corpus
- Citations where the paper doesn't support the claim

Only suggest edits you're confident about."""
