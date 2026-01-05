# Supervision Loops Proposal: Loops 2-5

This document explains the proposed supervision loops for the academic literature review workflow in plain English, discusses their value, and raises questions for clarification before implementation planning.

---

## Current Architecture Context

The workflow currently has **one supervision loop** (Loop 1: Theoretical Depth) that runs after the initial report synthesis. It works like this:

1. **Analyze**: Opus reads the complete report and identifies one theoretical gap (underlying theory, methodological foundation, unifying thread, or foundational concept)
2. **Expand**: If a gap is found, run a focused mini-research cycle (discovery → diffusion → processing) on that specific topic
3. **Integrate**: Opus weaves the new findings into the report
4. **Iterate**: Repeat up to `max_iterations` times, or exit early if no gaps remain

The architecture already supports adding more loops via a `supervision/loops/` directory structure and a planned registry pattern.

---

## Proposed Loop 2: Literature Base Expansion

### What It Does

After Loop 1 completes (theoretical depth is addressed), Loop 2 asks Opus to step back and consider: "Given this argument and the evidence marshalled, what **entire literature base** might be missing that would significantly add to or challenge the argument?"

This is different from Loop 1's gap-finding. Loop 1 looks for theoretical concepts to shore up; Loop 2 looks for entirely different bodies of scholarship that intersect with the topic.

**Example**: A literature review on "AI in medical diagnosis" might have thoroughly covered computer science and radiology literature, but Loop 2 might identify that:
- Health economics literature on diagnostic cost-effectiveness is absent
- Medical sociology on patient-physician trust with algorithmic tools is missing
- Legal scholarship on liability for AI misdiagnosis could challenge the implementation recommendations

For each missing literature base identified:
1. Run a full mini-review (same quality preset, no supervision loops) on that literature base
2. Integrate the findings into the main report
3. Check if more literature bases are needed

**Pass-through condition**: Opus determines the report already engages with all important literature bases.

### Why It Strengthens the Report

- **Interdisciplinary rigor**: Academic work is often criticized for staying within disciplinary silos. This loop actively seeks cross-disciplinary perspectives.
- **Anticipates criticism**: Reviewers often ask "did you consider X literature?" This loop preemptively addresses that.
- **Strengthens or challenges the argument**: Finding literature that challenges the argument is intellectually honest and makes the review more defensible.

### Questions to Clarify

1. **How many literature bases per iteration?** Your notes say "one new literature base per iteration" — is this to keep each integration manageable?

2. **What constitutes a "full mini-review"?** Same quality setting but no supervision loops — does this mean Loops 1-5 don't run on the mini-review, or just no recursive Loop 2?

3. **Integration strategy**: Should the new literature base get its own thematic section, or be woven throughout existing sections where relevant?

4. **Iteration cap**: Should this loop have its own max_iterations, or share with the overall supervision budget?

5. **Weighting for "challenge" literature**: Should the system actively prefer literature bases that might challenge the argument, or treat supportive and challenging literature equally?

---

## Proposed Loop 3: Structure and Cohesion Review

### What It Does

This is a **two-agent editorial pass**:

**Agent 1 (Structural Analyst)**:
- Reads the complete report
- Produces a structured edit manifest (not prose) containing:
  - Suggested section reordering (if needed)
  - Sections that should be merged or split
  - Transitions that need strengthening
  - Redundancies to consolidate
  - Logical flow issues
  - Narrative gaps where the argument doesn't connect

**Agent 2 (Editor)**:
- Receives the report + the structured edit manifest
- Executes all structural edits
- Returns the complete restructured document
- **Constraint**: No new content can be added — but can insert `[TODO: ...]` markers where more research or detail would improve the piece

### Why It Strengthens the Report

- **Separation of concerns**: Analysis and execution are cognitively different tasks. Separating them allows each agent to focus.
- **Auditability**: The structured manifest creates a record of what changes were made and why.
- **Catches drift**: Long documents written section-by-section often have structural issues that only become apparent when reading the whole. This loop reads holistically.
- **Preserves the TODO list**: By allowing TODOs but not fabrication, it maintains intellectual honesty while flagging areas for future work.

### Questions to Clarify

1. **Manifest format**: What structure for the edit instructions? I'm imagining something like:
   ```
   - reorder_sections: [{from: 3, to: 1}, ...]
   - merge_sections: [{sections: [2, 4], reason: "overlapping themes"}, ...]
   - add_transition: {after_section: 2, guidance: "bridge from X to Y concept"}
   - flag_redundancy: {locations: [...], keep: "section 3 version"}
   ```
   Does this match your vision?

2. **Wholesale reorganization**: You mention this is possible. Should there be guardrails (e.g., introduction must stay first, methodology before findings)?

3. **TODO format**: Should TODOs follow a specific format so they can be programmatically extracted later? e.g., `<!-- TODO: More detail needed on X -->`

4. **Iteration**: Is this a single-pass loop, or can it iterate (restructure → check → restructure again)?

---

## Proposed Loop 4: Section-Level Deep Editing

### What It Does

This is the **substantive editing pass** where content can actually be added:

**Phase A (Parallel Section Editing)**:
- The report (as restructured by Loop 3) is split into sections
- Each section goes to a dedicated Opus instance
- Each instance sees:
  - The full report (for context)
  - Its assigned section (for editing)
  - Access to research tools (OpenAlex, citation network)
  - Access to the document store (query tools for paper summaries, full texts)
- Task: Make detailed edits and additions. Fill in gaps. Strengthen arguments. Add nuance. Ensure claims are well-supported.

**Phase B (Holistic Review)**:
- Another Opus reads the complete reassembled report
- Can flag sections for re-editing (if `max_iterations > 1`)
- Focus: Does the whole piece cohere after the parallel edits? Are there now inconsistencies?

**Phase C (Optional Re-edit)**:
- Flagged sections return to their Opus instances for another pass
- This addresses the "parallel editing created inconsistencies" problem

**Constraints**:
- Minor grammatical/sentence-level edits can be left to Loop 5
- Focus is substantive: argument, evidence, interpretation

### Why It Strengthens the Report

- **Deep engagement**: Each section gets dedicated attention with full tool access
- **Parallel efficiency**: Sections can be edited simultaneously
- **Quality control**: The holistic reviewer catches problems that parallel editing can introduce
- **Tool access**: Editors can look up original papers, find additional evidence, verify claims

### Questions to Clarify

1. **Section granularity**: What counts as a "section"? Thematic sections only, or also introduction/methodology/discussion?

2. **Tool access scope**: Should section editors be able to add entirely new papers to the corpus, or only use existing papers more effectively?

3. **Re-edit triggering**: What criteria should the holistic reviewer use to flag sections? Specific structured checklist, or open-ended judgment?

4. **Coordination between sections**: If one section editor wants to reference something in another section, how is this handled? (Cross-references, shared findings)

5. **Loop 3 TODOs**: Should section editors specifically address the TODOs inserted by Loop 3?

6. **Edit tracking**: Should edits be tracked (before/after) for transparency, or just output the final version?

---

## Proposed Loop 5: Fact and Reference Checking

### What It Does

This is the **verification and polish pass**, using Haiku for efficiency:

**For each section** (with full report as context):

**Check A: Fact Checking**
- Verify factual claims against source documents
- Flag unsupported claims
- Check that interpretations of cited works are accurate
- Verify statistics, dates, names, terminology

**Check B: Reference Checking**
- Every citation points to a real paper in the corpus
- Cited papers actually support the claims made
- No orphan references (cited but not in bibliography)
- No missing citations (claims that should be cited but aren't)

**Output format** (per your specification):
```python
class Edit(BaseModel):
    find: str  # unique text to locate
    replace: str  # replacement text

class DocumentEdits(BaseModel):
    edits: list[Edit]
    reasoning: str
```

Edits are applied programmatically (like Anthropic's `str_replace` tool).

**Constraints**:
- No new TODOs can be added at this stage
- This is the final substantive pass
- Haiku-level model for cost efficiency

### Why It Strengthens the Report

- **Accuracy**: Catches misquotations, misattributions, and factual errors
- **Citation integrity**: Ensures every claim has appropriate support
- **Programmatic application**: The JSON edit format allows:
  - Audit trail of all changes
  - Deterministic application (no LLM interpretation of instructions)
  - Easy rollback if needed
  - Validation before application (check `find` strings exist and are unique)
- **Cost efficiency**: Haiku is much cheaper than Opus/Sonnet, and this task is well-suited to careful, focused checking rather than creative synthesis

### Questions to Clarify

1. **Fact vs. Reference separation**: You mention these should be separate checks. Two sequential passes per section, or two parallel passes that get merged?

2. **Tool access**: Same tools as Loop 4 (research tools + store queries) for verification?

3. **Handling ambiguity**: What if Haiku can't verify a claim (source document is unclear, claim is interpretive)? Flag for human review? Leave unchanged?

4. **Uniqueness of `find` strings**: What if the find string appears multiple times? Require longer context? Error and skip?

5. **Edit validation**: Should edits be validated before application (string exists, is unique, replacement is different from original)?

6. **Minor grammar**: You mention "minor grammatical and/or sentence-level edits." Does this mean Loop 5 also handles copy-editing, or is that a separate concern?

7. **No new TODOs**: But can existing TODOs from Loop 3 that weren't addressed in Loop 4 be removed, flagged, or left in place?

---

## My Suggestions

### 1. Loop Ordering and Dependencies

The proposed order (2 → 3 → 4 → 5) makes logical sense:
- **Loop 2** adds new content (literature bases)
- **Loop 3** restructures after new content is added
- **Loop 4** deepens each section
- **Loop 5** polishes and verifies

However, consider whether **Loop 3 should run after Loop 4** as well. Parallel section editing in Loop 4 might introduce structural issues that need another cohesion pass.

**Suggestion**: Consider a lightweight "cohesion check" after Loop 4, or allow Loop 5 to flag structural issues for human review.

### 2. Progress Tracking and Early Exit

Each loop should have clear exit conditions:
- **Loop 2**: "All important literature bases covered" (pass-through)
- **Loop 3**: Single pass by design? Or "no structural issues found"?
- **Loop 4**: "Holistic reviewer approves all sections" or max iterations
- **Loop 5**: "No fact/reference errors found" or all edits applied

**Suggestion**: Add a `LoopStatus` tracking object that records what each loop found and did, for debugging and transparency.

### 3. Cost Management

Running 5 Opus-heavy loops could be expensive. Consider:
- **Budget caps**: Per-loop and total token budgets
- **Early termination**: If report quality is already high, skip later loops
- **Quality preset integration**: Maybe "quick" preset skips Loops 3-4, "comprehensive" runs all

**Suggestion**: Add a `supervision_budget` setting separate from `max_stages`.

### 4. Loop 5 Edit Format Enhancement

The proposed `Edit` schema is minimal. Consider adding:
```python
class Edit(BaseModel):
    find: str
    replace: str
    edit_type: Literal["fact_correction", "citation_fix", "grammar", "clarity"]
    confidence: float  # how confident the model is this edit is correct
    source_doi: Optional[str]  # if fact correction, which paper supports this
```

This enables:
- Filtering edits by type
- Requiring human review for low-confidence edits
- Audit trail linking corrections to sources

### 5. Handling Conflicts Between Loops

What happens if:
- Loop 2 adds literature that contradicts Loop 1's integration?
- Loop 4's parallel editors make inconsistent claims?
- Loop 5 finds facts that contradict Loop 4's additions?

**Suggestion**: Each loop should have explicit conflict detection. When conflicts are found, either:
- Flag for human review
- Prefer more recent/authoritative sources
- Add explicit "there is debate on this point" language

### 6. Test Mode Considerations

For testing, you'll want to run individual loops in isolation.

**Suggestion**: Design each loop to be independently runnable with a "start from this report state" capability.

### 7. Transparency and Changelog

Academic integrity matters. Consider generating a "revision history" appendix that documents:
- What each loop changed
- Why (the supervisor's reasoning)
- What literature was added
- What claims were corrected

This could be invaluable for:
- Debugging the workflow
- Demonstrating rigor to reviewers
- Understanding how the final product evolved

---

## Summary Table

| Loop | Purpose | Agent(s) | Key Output | Exit Condition |
|------|---------|----------|------------|----------------|
| 1 (existing) | Theoretical depth | Opus × 3 | Gap-filled report | No gaps / max iterations |
| 2 (proposed) | Literature base expansion | Opus + mini-workflow | Cross-disciplinary report | All bases covered |
| 3 (proposed) | Structure & cohesion | Opus (analyst) + Opus (editor) | Restructured report + TODOs | Single pass |
| 4 (proposed) | Section deep editing | Opus × N (parallel) + Opus (reviewer) | Substantively edited report | Reviewer approves / max iterations |
| 5 (proposed) | Fact & reference checking | Haiku × N | JSON edits applied | All sections verified |

---

## Next Steps

Once the above questions are clarified, we can proceed to:
1. Design the data structures (extended `IdentifiedIssue`, loop-specific state)
2. Define the prompts for each agent role
3. Implement the loop infrastructure (registry, chaining, routing)
4. Implement each loop
5. Add quality preset integration
6. Test with representative cases

Please review this document and let me know:
- Which clarifications you'd like to address
- Any aspects I've misunderstood
- Whether my suggestions align with your vision
