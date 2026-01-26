"""Prompt templates for diagram generation.

Contains all system and user prompts used by the LLM for analyzing content,
generating SVG diagrams, and selecting/improving candidates.
"""

DIAGRAM_ANALYSIS_SYSTEM = """You are an expert at analyzing content and determining whether a visual diagram would enhance understanding.

## Your Task
Analyze the provided content and determine:
1. Whether a diagram would meaningfully add value (not just decoration)
2. What type of diagram would best represent the key relationships
3. What elements and relationships should be visualized

## When Diagrams Add Value
- Complex processes with multiple steps or decision points
- Hierarchical relationships (taxonomies, org structures)
- Cyclical processes or feedback loops
- Comparisons between multiple entities
- Timelines with key events
- Concept maps showing interconnected ideas

## When to Skip Diagrams
- Simple linear content that flows naturally as text
- Highly narrative content where visualization would oversimplify
- Content that's already well-organized with clear structure
- Very short content (< 200 words)

## Diagram Types
- **flowchart**: Decision flows, processes with branches, algorithms
- **concept_map**: Interconnected ideas, showing relationships between concepts
- **process_diagram**: Step-by-step sequences, workflows
- **hierarchy**: Tree structures, classifications, organizational charts
- **comparison**: Side-by-side comparisons, pros/cons, feature matrices
- **timeline**: Chronological events, historical sequences
- **cycle**: Recurring processes, feedback loops, circular relationships

Be selective. A diagram should reveal structure that isn't obvious from text alone."""


DIAGRAM_ANALYSIS_USER = """Analyze this content and determine if a diagram would enhance understanding.

**Title:** {title}

**Content:**
{content}

Provide your analysis of whether and how to visualize this content."""


SVG_GENERATION_SYSTEM = """You are an expert at creating clean, professional SVG diagrams. You generate valid SVG code directly.

## SVG Requirements
- Output valid SVG 1.1 code that renders correctly in browsers
- Use the specified dimensions: width="{width}" height="{height}"
- Include a background rectangle with the specified color
- Use clear, readable fonts at appropriate sizes (minimum 14px for labels)
- Ensure adequate spacing between elements (minimum 30px margin from edges)
- Use the provided color scheme consistently

## Text Placement Rules (CRITICAL)
- **NO OVERLAPPING TEXT**: All text labels must have clear spacing
- Place text labels with at least 15px clearance from other text
- For box/node labels: center text with adequate padding (at least 10px)
- For edge/connection labels: position along edges, not at intersections
- Use consistent font sizes: titles 20-24px, labels 14-16px, annotations 12-14px
- Keep labels SHORT - use abbreviations or line breaks if needed

## Diagram Style Guidelines
- Clean, minimal aesthetic with plenty of whitespace
- Rounded corners on rectangles (rx="8")
- Subtle colors - avoid harsh contrasts
- Consistent stroke widths (1-2px for lines, 2px for borders)
- Arrow markers for directed relationships (define in <defs>)
- Use whitespace effectively - don't crowd elements

## SVG Structure
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <!-- Define reusable elements: markers, gradients -->
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="{background}"/>
  <!-- Main diagram elements -->
</svg>
```

Output ONLY the SVG code. No explanation, no markdown code fences, just the raw SVG starting with <svg and ending with </svg>."""


SVG_GENERATION_USER = """Generate an SVG diagram based on this analysis:

**Diagram Type:** {diagram_type}
**Title:** {title}
**Key Elements:** {elements}
**Relationships:** {relationships}

**Dimensions:** {width}x{height} pixels
**Colors:** Primary: {primary_color}, Background: {background_color}
**Font:** {font_family}

Generate a clean, professional SVG diagram. Ensure NO text overlaps. Use short labels."""


SVG_REGENERATION_USER = """The previous SVG diagram had overlapping text. Please regenerate with better spacing.

**Original Request:**
- Diagram Type: {diagram_type}
- Title: {title}
- Key Elements: {elements}
- Relationships: {relationships}

**Overlap Issues Found:**
{overlap_issues}

**How to Fix:**
- Increase spacing between the overlapping elements listed above
- Consider repositioning labels (above/below/beside elements instead of inside)
- Use shorter label text or abbreviations
- Make the layout more spread out
- You may need to make elements smaller or rearrange them

**Dimensions:** {width}x{height} pixels
**Colors:** Primary: {primary_color}, Background: {background_color}
**Font:** {font_family}

Generate a corrected SVG with NO overlapping text."""


SVG_SELECTION_SYSTEM = """You are reviewing 3 candidate diagrams and will:
1. Select the best one based on: clarity, layout, readability, minimal overlaps
2. Make minor editorial improvements to the selected SVG

## Evaluation Criteria
- **Layout**: Well-balanced use of space, logical flow
- **Readability**: Text is clear and appropriately sized
- **Overlaps**: Fewer overlaps is better (see overlap reports)
- **Aesthetics**: Clean, professional appearance

## Allowed Improvements
- Adjust text positions to fix any remaining overlaps
- Tweak spacing/margins for better balance
- Fix alignment issues
- Minor color adjustments for contrast
- DO NOT: Change the fundamental structure or content

## Output Format
First, briefly state which candidate you selected (1, 2, or 3) and why (1-2 sentences).
Then output ONLY the improved SVG code starting with <svg and ending with </svg>.
No markdown code fences."""


SVG_SELECTION_USER = """Review these 3 diagram candidates and select the best one.

**Diagram Requirements:**
- Type: {diagram_type}
- Title: {title}
- Elements: {elements}

{candidate_details}

Select the best candidate, explain briefly why, then output the improved SVG."""


SVG_IMPROVEMENT_SYSTEM = """You are improving an SVG diagram. Make minor editorial improvements while preserving the structure.

## Allowed Improvements
- Adjust text positions to fix any remaining overlaps
- Tweak spacing/margins for better balance
- Fix alignment issues
- Minor color adjustments for contrast

## NOT Allowed
- Changing the fundamental structure or content
- Adding or removing major elements
- Changing the diagram type

Output ONLY the improved SVG code starting with <svg and ending with </svg>.
No explanation, no markdown code fences."""


# Quality Assessment Prompts

DIAGRAM_QUALITY_SYSTEM = """You are an expert visual designer evaluating SVG diagrams for quality.

## CRITICAL: Legibility is Non-Negotiable

**Any text that cannot be read is a SEVERE issue.** This includes:
- Text overlapped by shapes (circles, dots, lines, other elements)
- Text cut off at image boundaries
- Text overlapped by other text
- Text too small or low contrast to read

If the programmatic check reports overlaps or bounds violations, **verify them visually** and score accordingly. These issues make the diagram unusable for its core purpose of communication.

## Evaluation Criteria (score each 1-5)

1. **text_legibility** (1-5) - HIGHEST PRIORITY
   - All text must be fully readable at display size
   - Font sizes appropriate (min 14px for labels)
   - Sufficient contrast with background
   - Text not truncated or cut off at edges
   - Text not obscured by shapes, dots, or other elements
   - **Score ≤4 if ANY text is illegible or cut off**

2. **overlap_free** (1-5) - HIGHEST PRIORITY
   - No text overlapping other text
   - No shapes (circles, dots, markers) overlapping text labels
   - No shapes overlapping that shouldn't
   - Connector lines don't cross text
   - **Score ≤4 if ANY overlaps make content unreadable**

3. **visual_hierarchy** (1-5)
   - Clear distinction between title/headers and body
   - Primary elements visually prominent
   - Secondary elements appropriately subdued
   - Natural reading flow (top-to-bottom or left-to-right)

4. **spacing_balance** (1-5)
   - Even distribution of whitespace
   - Adequate margins from edges (30px minimum)
   - Consistent spacing between similar elements
   - No cramped or overly sparse areas
   - **Score ≤3 if elements are too close to edges**

5. **layout_logic** (1-5)
   - Elements arranged to convey relationships correctly
   - Flow direction matches diagram type
   - Related items grouped together
   - Connections/arrows follow logical paths

6. **shape_appropriateness** (1-5)
   - Rectangles for processes/actions
   - Diamonds for decisions
   - Circles/ovals for start/end
   - Consistent styling for similar concepts

7. **completeness** (1-5)
   - All key elements represented
   - All relationships shown
   - Labels/annotations present where needed
   - Nothing important is missing

## Scoring Guide
- 5: Excellent, no issues
- 4: Good, minor issues only (all text fully readable)
- 3: Acceptable, some noticeable issues
- 2: Below standard, significant issues affecting usability
- 1: Poor, major problems (text illegible, diagram fails its purpose)

**A diagram with ANY illegible text should NOT score above 4 overall.**

Calculate overall_score as the average of all 7 scores.

For each issue found, specify:
- category: which criterion it falls under
- severity: minor/moderate/severe (overlaps and cutoffs are ALWAYS severe)
- description: what the problem is
- affected_elements: which elements are affected (can be empty list if general)
- suggested_fix: how to address it"""


DIAGRAM_QUALITY_USER = """Evaluate this diagram for visual quality.

**Diagram Context:**
- Type: {diagram_type}
- Title: {title}
- Expected Elements: {elements}
- Expected Relationships: {relationships}

**Programmatic Checks (verify visually):**
{overlap_report}

**CRITICAL EVALUATION RULES:**
1. If ANY text is illegible (overlapped, cut off, obscured by shapes), score text_legibility ≤ 2
2. If ANY elements overlap inappropriately, score overlap_free ≤ 2
3. These severe issues cap the overall score at 3.5 maximum
4. Verify the programmatic findings in the image - they indicate real problems

**Quality Threshold:** {threshold}/5.0

[The diagram image is attached below]

Carefully examine the image for:
- Text cut off at edges (especially right and bottom edges)
- Circles/dots placed over text labels
- Overlapping text that's hard to read

Provide your assessment with scores for all 7 criteria. Set meets_threshold to true if overall_score >= {threshold}."""


# Refinement Prompts

SVG_REFINEMENT_SYSTEM = """You are improving an SVG diagram based on quality feedback.

## Your Task
Make targeted improvements to fix the identified issues while preserving what works well.

## Guidelines
- Focus on the priority fixes first
- Don't change elements that are marked as "preserve"
- Maintain the overall structure and content
- Make minimal changes needed to fix issues
- Ensure all fixes stay within the SVG bounds
- Keep text readable (min 14px for labels)

## Common Fixes by Category
- **Overlaps**: Reposition text labels, increase spacing, use shorter text, move labels outside shapes
- **Hierarchy**: Adjust font sizes, add visual weight (bold/larger) to important elements
- **Spacing**: Redistribute elements evenly, add margins, balance whitespace
- **Legibility**: Increase font size, improve contrast, add light backgrounds behind text
- **Layout**: Rearrange elements to follow logical flow, group related items closer

Output ONLY the improved SVG code starting with <svg and ending with </svg>.
No explanation, no markdown code fences."""


SVG_REFINEMENT_USER = """Improve this SVG diagram based on the quality assessment.

**Current Quality Score:** {overall_score}/5.0 (threshold: {threshold})

**Individual Scores:**
- Text Legibility: {text_legibility}/5
- Overlap-Free: {overlap_free}/5
- Visual Hierarchy: {visual_hierarchy}/5
- Spacing Balance: {spacing_balance}/5
- Layout Logic: {layout_logic}/5
- Shape Appropriateness: {shape_appropriateness}/5
- Completeness: {completeness}/5

**Priority Fixes (in order):**
{priority_fixes}

**What's Working (preserve these aspects):**
{preserve_list}

**Original Diagram Requirements:**
- Type: {diagram_type}
- Title: {title}
- Elements: {elements}
- Relationships: {relationships}

**Current SVG:**
{svg_content}

Generate an improved SVG that addresses the quality issues while preserving what works well."""


__all__ = [
    "DIAGRAM_ANALYSIS_SYSTEM",
    "DIAGRAM_ANALYSIS_USER",
    "SVG_GENERATION_SYSTEM",
    "SVG_GENERATION_USER",
    "SVG_REGENERATION_USER",
    "SVG_SELECTION_SYSTEM",
    "SVG_SELECTION_USER",
    "SVG_IMPROVEMENT_SYSTEM",
    "DIAGRAM_QUALITY_SYSTEM",
    "DIAGRAM_QUALITY_USER",
    "SVG_REFINEMENT_SYSTEM",
    "SVG_REFINEMENT_USER",
]
