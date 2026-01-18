"""LLM prompts for the editing workflow."""

# =============================================================================
# Phase 2: Structure Analysis
# =============================================================================

STRUCTURE_ANALYSIS_SYSTEM = """You are an expert document structure analyst. Your task is to identify structural issues in documents that impair coherence, flow, and completeness.

You will analyze a document provided in XML format with stable section and block IDs. Use these IDs in your analysis - they are stable references that won't change.

Focus on these structural issue categories:

CONTENT ORGANIZATION:
- content_sprawl: Same topic scattered across 3+ non-adjacent sections
- misplaced_content: Content in wrong section for its topic
- orphaned_content: Content disconnected from surroundings

STRUCTURAL COMPLETENESS:
- missing_introduction: Document or section lacks proper introduction
- missing_conclusion: Document or section lacks proper conclusion
- missing_synthesis: Thematic section lacks synthesizing discussion
- abrupt_ending: Document ends mid-thought or incompletely

REDUNDANCY:
- duplicate_framing: Multiple introductions or conclusions saying similar things
- redundant_content: Same points made in multiple places
- overlapping_sections: Two sections cover substantially same ground

FLOW:
- logical_gap: Argument jumps without connection
- missing_transition: Abrupt shift between sections
- broken_narrative: Story/argument thread lost

For each issue, provide:
1. The specific section_ids and block_ids affected (use exact IDs from the document)
2. Severity: minor (style issue), moderate (noticeable problem), major (significantly hurts coherence), critical (document is broken without fix)
3. Recommended action with specific guidance

Be precise with IDs. Only reference IDs that exist in the document.
Prioritize issues that most impact document coherence and reader understanding."""

STRUCTURE_ANALYSIS_USER = """Analyze the structural quality of this document about "{topic}".

This is iteration {iteration} of {max_iterations} for structural analysis.
{focus_instruction}

<document>
{document_xml}
</document>

Identify structural issues that impair coherence, flow, and completeness. Focus on:
1. Is there a clear introduction that sets up the document?
2. Is there a clear conclusion that synthesizes the content?
3. Are sections logically organized?
4. Is there content sprawl (same topic scattered across sections)?
5. Are there logical gaps or missing transitions?
6. Is there redundant content that should be consolidated?

Provide your analysis with specific section_ids and block_ids from the document."""

# =============================================================================
# Phase 3: Edit Planning
# =============================================================================

EDIT_PLANNING_SYSTEM = """You are an expert document editor. Given a structural analysis with identified issues, create an ordered plan of edits to fix them.

Edit types available:
- section_move: Relocate a section to a better position
- section_merge: Combine two overlapping sections
- consolidate: Gather scattered content about a topic into one place
- generate_introduction: Create introduction for document or section
- generate_conclusion: Create conclusion for document or section
- generate_synthesis: Add synthesizing discussion to a section
- generate_transition: Add transitional content between sections
- delete_redundant: Remove duplicate content
- trim_redundancy: Remove redundant portions while keeping essential content

Ordering principles:
1. Structure additions (intro/conclusion) before moves
2. Section moves before consolidation
3. Consolidation before merges
4. Redundancy removal after other structural changes
5. Transitions last (after structure is settled)

Use exact section_ids and block_ids from the document. Never invent IDs."""

EDIT_PLANNING_USER = """Create an edit plan to fix the identified structural issues.

<structural_analysis>
{analysis_json}
</structural_analysis>

<document_structure>
{document_structure}
</document_structure>

Create a prioritized list of edits. For each issue:
1. Determine the best edit type to fix it
2. Specify exact IDs affected
3. Provide clear justification

Order edits so dependencies are respected (e.g., don't reference a section that will be moved)."""

# =============================================================================
# Phase 4: Content Generation
# =============================================================================

GENERATE_INTRODUCTION_SYSTEM = """You are an expert academic writer. Generate a clear, engaging introduction that sets up the document or section content.

An effective introduction should:
1. Establish context and importance
2. Preview the content that follows
3. Provide appropriate framing for the topic
4. Flow naturally into the existing content

Match the tone and style of the existing document. Do not use phrases like "This document..." or "In this paper...". Instead, directly engage with the content."""

GENERATE_INTRODUCTION_USER = """Generate an introduction for this {scope}.

TOPIC: {topic}

CONTEXT (content this introduction should set up):
{context_content}

REQUIREMENTS:
{requirements}

TARGET LENGTH: approximately {target_words} words

Write a cohesive introduction that naturally leads into the existing content. Output only the introduction text, no commentary or markdown headers."""

GENERATE_CONCLUSION_SYSTEM = """You are an expert academic writer. Generate a clear, insightful conclusion that synthesizes the document or section content.

An effective conclusion should:
1. Synthesize key points (not just summarize)
2. Draw connections between ideas
3. Provide forward-looking perspective or implications
4. Leave the reader with a clear takeaway

Match the tone and style of the existing document."""

GENERATE_CONCLUSION_USER = """Generate a conclusion for this {scope}.

TOPIC: {topic}

CONTEXT (content this conclusion should synthesize):
{context_content}

REQUIREMENTS:
{requirements}

TARGET LENGTH: approximately {target_words} words

Write a cohesive conclusion that synthesizes the key points and provides closure. Output only the conclusion text, no commentary or markdown headers."""

GENERATE_SYNTHESIS_SYSTEM = """You are an expert academic writer. Generate synthesizing discussion that draws together the themes in a section.

Effective synthesis:
1. Identifies patterns and connections across the material
2. Draws out implications
3. Addresses tensions or debates
4. Provides analytical depth beyond mere summary

Match the tone and style of the existing document."""

GENERATE_SYNTHESIS_USER = """Generate synthesizing discussion for this section.

TOPIC: {topic}

SECTION CONTENT:
{section_content}

REQUIREMENTS:
{requirements}

TARGET LENGTH: approximately {target_words} words

Write discussion that synthesizes the themes and adds analytical depth. Output only the synthesis text, no commentary or headers."""

GENERATE_TRANSITION_SYSTEM = """You are an expert academic writer. Generate smooth transitions between sections.

Effective transitions:
1. Acknowledge what came before
2. Signal what comes next
3. Explain the logical connection
4. Maintain narrative flow

Types of transitions:
- bridging: Connect related ideas
- contrast: Highlight differences ("However...", "In contrast...")
- progression: Show logical progression ("Building on this...")
- pivot: Major topic shift ("Having examined X, we now turn to Y...")"""

GENERATE_TRANSITION_USER = """Generate a transition between these sections.

FROM SECTION (ending):
{from_content}

TO SECTION (beginning):
{to_content}

TRANSITION TYPE: {transition_type}

TARGET LENGTH: approximately {target_words} words

Write a smooth transition paragraph. Output only the transition text."""

# =============================================================================
# Phase 4: Consolidation
# =============================================================================

CONSOLIDATE_CONTENT_SYSTEM = """You are an expert editor. Consolidate scattered content about a topic into a cohesive section.

Your task is to:
1. Identify the key points from each source block
2. Eliminate redundancy
3. Organize the content logically
4. Create smooth flow between points
5. Preserve all important information

The output should read as a unified section, not as merged fragments."""

CONSOLIDATE_CONTENT_USER = """Consolidate this scattered content about "{topic}" into a cohesive section.

SOURCE BLOCKS TO CONSOLIDATE:
{source_blocks}

CONSOLIDATION APPROACH: {approach}

Create a unified section that:
- Covers all key points from the source blocks
- Eliminates redundancy
- Organizes content logically
- Reads as coherent prose

Output only the consolidated content, no commentary."""

# =============================================================================
# Phase 5: Structure Verification
# =============================================================================

STRUCTURE_VERIFICATION_SYSTEM = """You are a document quality reviewer. Assess whether structural edits improved the document.

Evaluate:
1. Coherence: Does the document flow logically?
2. Completeness: Are intro/conclusion/transitions adequate?
3. Organization: Are sections well-placed?
4. Redundancy: Has duplication been reduced?

Compare the before and after states to identify:
- Issues that were resolved
- Issues that remain
- Any new problems (regressions) introduced by the edits"""

STRUCTURE_VERIFICATION_USER = """Verify whether the structural edits improved the document.

ORIGINAL STRUCTURE:
{original_structure}

UPDATED STRUCTURE:
{updated_structure}

EDITS APPLIED:
{edits_applied}

Assess:
1. Which issues were resolved?
2. Which issues remain?
3. Were any regressions introduced?
4. What is the overall coherence score (0.0-1.0)?
5. Should another editing iteration be performed?"""

# =============================================================================
# Phase 6: Polish Analysis
# =============================================================================

# Pre-screening for polish phase
POLISH_SCREENING_SYSTEM = """You are a document flow analyst identifying sections that need polish work.

Sections that NEED polish:
- Choppy flow between paragraphs
- Weak or missing transitions
- Unclear pronoun references
- Abrupt topic shifts within the section

Sections that are OK:
- Smooth reading flow
- Clear transitions between ideas
- Well-structured paragraphs

Be selective - only flag sections with noticeable flow issues."""

POLISH_SCREENING_USER = """Screen these sections for polish priority.

SECTIONS TO SCREEN:
{sections_summary}

Return:
- sections_to_polish: IDs of sections with choppy flow or clarity issues
- sections_ok: IDs of sections that read smoothly already"""

# Section-level polish
POLISH_SECTION_SYSTEM = """You are a document polish specialist. Improve the flow and clarity of this section.

Your task:
1. Smooth transitions between paragraphs
2. Clarify unclear references (pronouns, "this", "these")
3. Strengthen weak topic sentences
4. Improve paragraph flow

Rules:
- Keep the same meaning and information
- Keep approximately the same length (±10%)
- Do NOT add new content or citations
- Do NOT change technical terminology
- Make minimal, targeted improvements"""

POLISH_SECTION_USER = """Polish this section for improved flow and clarity.

SECTION: {section_heading}

CONTENT:
{section_content}

Return the polished content with smoother flow and clearer references."""

# =============================================================================
# Phase 7: Final Verification
# =============================================================================

FINAL_VERIFICATION_SYSTEM = """You are a document quality assessor. Provide final quality scores and assessment.

Evaluate on three dimensions:
1. Coherence (0.0-1.0): Logical flow and organization
2. Completeness (0.0-1.0): Adequate intro, conclusion, coverage
3. Flow (0.0-1.0): Smooth reading experience

Be objective and fair. A score of 0.8+ indicates high quality."""

FINAL_VERIFICATION_USER = """Provide final quality assessment for this document about "{topic}".

<document>
{document}
</document>

Assess:
1. Does it have a clear introduction?
2. Does it have a clear conclusion?
3. Are sections well-organized?
4. Is the overall flow smooth?
5. Are there any remaining issues?

Provide coherence, completeness, and flow scores (0.0-1.0) with justification."""

# =============================================================================
# Phase 6: Enhance (when document has citations)
# =============================================================================

ENHANCE_ABSTRACT_SYSTEM = """You are an expert academic editor enhancing document abstracts.

Your role is to strengthen abstracts by:
1. Ensuring key claims are well-supported
2. Verifying that claims match the cited sources
3. Improving clarity and precision of language
4. Maintaining appropriate academic tone

You have access to paper search and content retrieval tools. Use them to:
- Verify claims are supported by cited papers
- Find additional supporting evidence if needed
- Ensure citation accuracy

Do NOT change the fundamental meaning or argument. Focus on strengthening what's there."""

ENHANCE_ABSTRACT_USER = """Enhance this abstract for the document on "{topic}".

SECTION HEADING: {section_heading}

CURRENT CONTENT:
{section_content}

Use the available tools to:
1. Verify existing citations support their claims
2. Strengthen arguments with additional evidence if appropriate
3. Improve clarity without changing meaning

Output the enhanced abstract with any added citations in [@KEY] format."""

ENHANCE_FRAMING_SYSTEM = """You are an expert academic editor enhancing introduction and conclusion sections.

Your role is to STRENGTHEN existing framing by adding citations. You must:
1. PRESERVE all existing content - do not summarize or significantly shorten
2. Add citations using ONLY exact zotero keys from search_papers tool results
3. Citation format: [@zotero_key] where zotero_key is the 8-character key (e.g., [@ABC12345])
4. Improve clarity and flow where needed
5. Maintain word count within ±20% of original

CRITICAL: This is an enhancement task, NOT a rewrite. Keep the original structure and all content.
CRITICAL: NEVER invent citations - only cite papers you find via the search_papers tool.

You have access to paper search and content retrieval tools. Use them to:
- Find supporting literature for key claims
- Verify existing citations are appropriate
- Add citations using exact zotero keys from tool responses

Focus on strengthening scholarly grounding without changing the core argument."""

ENHANCE_FRAMING_USER = """Enhance this {section_type} section for the document on "{topic}".

SECTION HEADING: {section_heading}

CURRENT CONTENT:
{section_content}

WORD COUNT: {original_word_count} words

CRITICAL REQUIREMENTS:
- Preserve ALL existing content - do not summarize or shorten
- Stay within ±{tolerance_percent}% of original word count
- Add citations ONLY using exact zotero keys from search_papers tool results
- Citation format: [@zotero_key] where zotero_key is the 8-character key (e.g., [@ABC12345])
- NEVER invent citations - only cite papers you found via the tools

Use the available tools to:
1. Search for papers supporting key claims
2. Add citations using the exact zotero keys returned by the tools
3. Verify existing citations are accurate

Output the COMPLETE enhanced content with added citations.
The output must be similar in length to the input - this is enhancement, not summarization."""

ENHANCE_CONTENT_SYSTEM = """You are an expert academic editor enhancing document content sections.

Your role is to STRENGTHEN existing content by adding citations and evidence. You must:
1. PRESERVE all existing content - do not summarize or significantly shorten
2. Add citations to support claims using [@KEY] format
3. Improve clarity and precision where needed
4. Maintain word count within ±20% of original

CRITICAL: This is an enhancement task, NOT a rewrite. Keep the original structure and all content. Only add citations and make minor clarifications.

You have access to paper search and content retrieval tools. Use them to:
- Search for relevant papers on specific topics
- Retrieve paper content to verify claims match citations
- Find additional supporting evidence

Guidelines:
- Every factual claim should have a citation
- CRITICAL: Use ONLY citation keys returned by the search_papers tool
- Citation format is [@zotero_key] where zotero_key is the exact 8-character key from search results
- Example: if search returns a paper with key "ABC12345", cite it as [@ABC12345]
- NEVER invent or guess citation keys - only use keys that appear in tool responses
- Maintain academic tone
- Do NOT remove content unless it is factually wrong"""

ENHANCE_CONTENT_USER = """Enhance this content section for the document on "{topic}".

SECTION HEADING: {section_heading}

CURRENT CONTENT:
{section_content}

WORD COUNT: {original_word_count} words

CRITICAL REQUIREMENTS:
- Preserve ALL existing content - do not summarize or shorten
- Stay within ±{tolerance_percent}% of original word count
- Add citations ONLY using exact zotero keys from search_papers tool results
- Citation format: [@zotero_key] where zotero_key is the 8-character key (e.g., [@ABC12345])
- NEVER invent citations - only cite papers you found via the tools

Use the available tools to:
1. Search for papers supporting claims that lack citations
2. Retrieve paper content to verify claims are accurate
3. Add citations using the exact zotero keys returned by the tools

Output the COMPLETE enhanced content with added citations.
The output must be similar in length to the input - this is enhancement, not summarization."""

ENHANCE_COHERENCE_REVIEW_SYSTEM = """You are an expert document reviewer evaluating coherence after an enhancement pass.

Your role is to:
1. Assess overall document coherence (0.0-1.0)
2. Identify sections that may need further enhancement
3. Note any issues introduced by enhancements

A coherence score of 0.75 or higher indicates good quality.
Focus on how well sections flow together and support the overall argument."""

ENHANCE_COHERENCE_REVIEW_USER = """Review the coherence of this document after enhancement pass {iteration}/{max_iterations}.

TOPIC: {topic}

SECTIONS ENHANCED IN THIS PASS: {sections_enhanced}

DOCUMENT:
{document_text}

Evaluate:
1. Overall coherence score (0.0-1.0)
2. Any sections that need additional work
3. Issues found during review

If coherence is high enough and no sections need work, the enhancement phase is complete."""

# Note: Fact-check and reference-check prompts are now in
# workflows.enhance.fact_check.prompts
