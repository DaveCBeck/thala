"""Prompts for Opus one-by-one integration producing synthesized documents."""

INTEGRATION_SYSTEM = """You are integrating research findings from {language_name} into an evolving English synthesis document.

Context:
- You are building a comprehensive, unified research review that incorporates multilingual sources
- Previous languages have already been integrated (see current document)
- Your task: thoughtfully integrate {language_name} findings

Guidelines:
1. ADD new insights not already covered
2. ENHANCE existing sections with supporting or contrasting evidence
3. NOTE where {language_name} sources provide unique perspectives (use inline notes like "[German sources add: ...]")
4. MAINTAIN academic tone and proper attribution
5. FLAG any conflicts with existing content for reader awareness
6. PRESERVE the document's coherent narrative flow

Do NOT simply append content. Weave it in naturally where it fits thematically.

Output:
1. The updated synthesis document
2. Notes on what was added/changed"""

INTEGRATION_USER = """Current synthesis document:
{current_document}

---

Sonnet's guidance on {language_name}'s unique contributions:
{sonnet_guidance}

---

{language_name} findings to integrate:
{language_findings}

---

Integrate these findings into the synthesis document. The result should read as one coherent document, not a patchwork."""

INITIAL_SYNTHESIS_SYSTEM = """You are creating the initial synthesis document from English-language research findings.

This document will serve as the foundation that other languages' findings will be integrated into.

Create a well-structured research synthesis that:
1. Has clear sections organized by theme/topic
2. Includes an Executive Summary
3. Has proper citations and source attributions
4. Leaves natural places where additional perspectives could enhance the content
5. Is written in clear, academic English

This is the STARTING POINT - it will be enriched with non-English sources in subsequent steps."""

INITIAL_SYNTHESIS_USER = """Topic: {topic}

Research questions:
{research_questions}

English-language research findings:
{english_findings}

Create the initial synthesis document that will serve as the foundation for multi-language integration."""

FINAL_ENHANCEMENT_SYSTEM = """You are finalizing a multi-language research synthesis document.

The document has been built by integrating findings from multiple languages. Your task is to:

1. ADD an "Enhanced Coverage Notes" section at the end summarizing how non-English sources improved the research
2. ENSURE the document flows coherently as one unified piece
3. FIX any awkward transitions between integrated sections
4. ADD a "Methodology" section listing languages analyzed and workflows used
5. POLISH the Executive Summary to reflect the full multi-language scope

Do not remove content - only enhance, smooth, and finalize."""

FINAL_ENHANCEMENT_USER = """Current synthesis document:
{current_document}

Languages integrated: {languages_list}
Workflows used: {workflows_list}

Integration notes from each language:
{integration_notes}

Finalize this document for delivery."""
