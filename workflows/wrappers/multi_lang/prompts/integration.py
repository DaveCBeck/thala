"""Prompts for Opus one-by-one integration producing synthesized documents."""

INTEGRATION_SYSTEM = """You are integrating research findings from {language_name} into an evolving English synthesis document.

Context:
- You are building a comprehensive, unified research review
- Previous languages have already been integrated (see current document)
- Your task: thoughtfully integrate {language_name} findings into the content

Guidelines:
1. ADD new insights not already covered
2. ENHANCE existing sections with supporting or contrasting evidence
3. PRESERVE ALL citation keys in [@KEY] format - these are Zotero references that must remain intact
4. When mentioning contributions from {language_name} sources, briefly note the language origin in parentheses, e.g., "(from Spanish sources)"
5. MAINTAIN academic tone with proper citations
6. PRESERVE the document's coherent narrative flow
7. Focus on CONTENT, not process - do not add methodology sections or meta-commentary about the integration

CRITICAL: Citation keys like [@7NM5HWY5] or [@ABC123] MUST be preserved exactly as they appear. These link to the reference database.

Do NOT simply append content. Weave it in naturally where it fits thematically.

Output:
1. The updated synthesis document with all citations preserved
2. Brief notes on what was added/changed"""

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
3. PRESERVES ALL citation keys in [@KEY] format exactly as they appear - these are Zotero references
4. Leaves natural places where additional perspectives could enhance the content
5. Is written in clear, academic English
6. Focuses on CONTENT, not methodology or process

CRITICAL: Citation keys like [@7NM5HWY5] MUST be preserved exactly. Do not describe them, include them inline.

This is the STARTING POINT - it will be enriched with non-English sources in subsequent steps."""

INITIAL_SYNTHESIS_USER = """Topic: {topic}

Research questions:
{research_questions}

English-language research findings:
{english_findings}

Create the initial synthesis document that will serve as the foundation for multi-language integration."""

FINAL_ENHANCEMENT_SYSTEM = """You are finalizing a multi-language research synthesis document.

The document has been built by integrating findings from multiple languages. Your task is to:

1. ENSURE the document flows coherently as one unified piece
2. FIX any awkward transitions between integrated sections
3. POLISH the Executive Summary to reflect the full scope of findings
4. PRESERVE ALL citation keys in [@KEY] format exactly as they appear
5. End the document after the Conclusion - do not add methodology sections, coverage notes, or other meta-commentary

CRITICAL:
- Citation keys like [@7NM5HWY5] MUST be preserved exactly
- Focus on CONTENT quality, not process documentation
- The document should read as a focused research synthesis, not a report about the synthesis process

Do not remove substantive content. Remove any meta-commentary about the integration process."""

FINAL_ENHANCEMENT_USER = """Current synthesis document:
{current_document}

Languages integrated: {languages_list}
Workflows used: {workflows_list}

Integration notes from each language:
{integration_notes}

Finalize this document for delivery."""
