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


__all__ = [
    "DIAGRAM_ANALYSIS_SYSTEM",
    "DIAGRAM_ANALYSIS_USER",
    "SVG_GENERATION_SYSTEM",
    "SVG_GENERATION_USER",
    "SVG_REGENERATION_USER",
    "SVG_SELECTION_SYSTEM",
    "SVG_SELECTION_USER",
    "SVG_IMPROVEMENT_SYSTEM",
]
