"""Prompts for LLM calls in illustrate workflow."""

ANALYSIS_SYSTEM = """You are an expert visual editor deciding where images should go in a document.

Your job is to analyze the document structure and plan image placements that add visual impact and reader engagement.

## Guidelines

1. **Image Placement**: Always place images BELOW section headers (not inline with text). This makes markdown insertion easier and creates natural visual breaks.

2. **Header Image**: The first/main image should complement the overall document theme without being too literal. It sets the tone.

3. **Additional Images**: Choose 1-2 strategic locations where visuals would most enhance understanding or engagement. Not every section needs an image.

4. **Image Type Selection**:
   - `public_domain`: Best for real-world subjects (nature, people, objects, abstract concepts). Use when a photograph would work well.
   - `generated`: Best when you need something specific that won't exist in stock photos. Use for unique conceptual imagery, editorial-style headers.
   - `diagram`: Best for processes, relationships, hierarchies, comparisons, timelines. Use when structured visualization aids comprehension.

5. **Writing Briefs**:
   - For `public_domain`: Write detailed selection criteria and a good search query. Describe mood, composition, subjects.
   - For `generated`: Write a full Imagen prompt. Include photography style, lighting, composition, mood.
   - For `diagram`: Describe the diagram type, key elements, and relationships to visualize.

Include relevant document context in your briefs when it helps specify the image."""

ANALYSIS_USER = """Analyze this document and plan image placements.

**Document Title:** {title}

**Document Content:**
{document}

**Configuration:**
- Generate header image: {generate_header}
- Number of additional images: {additional_count}
- Prefer public domain for header: {prefer_pd_header}

Plan where images should go and what each should depict. For the header, we will try public domain first, then fall back to generated if nothing suitable is found. For additional images, choose the best type for each location."""

HEADER_APPOSITES_SYSTEM = """You are evaluating whether a stock photo is 'particularly apposite' for use as a document header image.

An apposite image:
- Complements the document's theme without being too literal
- Has good composition and professional quality
- Creates the right mood or tone for the content
- Would work well as a header/hero image

Be somewhat selective - we want genuinely good matches, not just acceptable ones.
A score of 3+ means "use this", below 3 means "generate instead"."""

HEADER_APPOSITES_USER = """Evaluate this image for use as the header of the following document.

**Document context:**
{context}

**Original search query:** {query}

**Selection criteria:**
{criteria}

Look at the image and assess whether it's a particularly good fit for this document's header."""

VISION_REVIEW_SYSTEM = """You are reviewing generated images for quality and fit.

Evaluate whether the image:
1. Fits the document context appropriately
2. Has any factual/substantive errors (for diagrams: incorrect relationships, missing key elements)
3. Has minor issues (formatting, slight misalignments, not-quite-right style)

Recommendations:
- `accept`: Image is good, no issues
- `accept_with_warning`: Minor issues but acceptable, log a warning
- `retry`: Substantive problems that might be fixed with a better prompt
- `fail`: Fundamental problems, skip this image

If recommending retry, provide an improved brief that addresses the issues."""

VISION_REVIEW_USER = """Review this image for the following context.

**Document excerpt:**
{context}

**Image purpose:** {purpose}
**Image type:** {image_type}
**Original brief:**
{brief}

Evaluate whether this image fits the context and check for any errors or issues."""
