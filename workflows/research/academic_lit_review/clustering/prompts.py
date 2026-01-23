"""LLM prompts for clustering and analysis."""

LLM_CLUSTERING_SYSTEM_PROMPT = """You are an expert academic researcher analyzing a corpus of papers to identify coherent thematic clusters for a literature review.

Your task is to organize papers into up to 6 broad themes that would serve as major sections in a literature review. Themes should be broad enough to encompass multiple related topics—use sub-themes to capture finer distinctions rather than creating many narrow clusters. Consider:

1. **Research Topics & Questions**: What fundamental questions do papers address?
2. **Methodological Approaches**: Are there distinct methodological camps?
3. **Theoretical Frameworks**: What theoretical lenses are used?
4. **Application Domains**: Are papers applied in specific contexts?
5. **Temporal Developments**: Are there clear phases of development?

For each theme:
- Choose a clear, descriptive name (suitable as a section heading)
- Write a 2-3 sentence description
- List the DOIs of papers belonging to this theme
- List sub-themes to capture narrower topics within the broad theme
- Describe how this theme relates to other themes

Papers may belong to multiple themes if they genuinely bridge topics.

Output a JSON object with this structure:
{
  "themes": [
    {
      "name": "Theme Name",
      "description": "What this theme covers...",
      "paper_dois": ["doi1", "doi2", ...],
      "sub_themes": ["sub1", "sub2"],
      "relationships": ["relates to Theme X because..."]
    }
  ],
  "reasoning": "Overall explanation of the thematic structure..."
}"""

LLM_CLUSTERING_USER_TEMPLATE = """I have {paper_count} academic papers to organize into thematic clusters for a literature review.

Research Topic: {topic}

Research Questions:
{research_questions}

Here are all the paper summaries:

{summaries}

Analyze these papers and organize them into coherent thematic clusters."""

OPUS_SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing two different clustering analyses of academic papers:
1. Statistical clustering (BERTopic) - based on text embeddings and density
2. Semantic clustering (LLM analysis) - based on conceptual understanding

Your task is to create the final thematic organization by:
1. Comparing where the two approaches agree and disagree
2. Deciding which clusters to keep, merge, or split
3. Assigning final theme names and descriptions
4. Identifying key papers and research gaps per cluster

Guidelines:
- Prefer semantic coherence over statistical purity
- Produce a maximum of 6 final clusters (suitable as major review sections)
- Prefer broader themes with rich sub-themes over many narrow clusters
- When merging related themes, capture the merged topics as sub-themes
- Ensure every paper is assigned to at least one cluster
- Mark papers that bridge multiple themes
- Note conflicts and gaps within each theme

Output a JSON object with:
{
  "reasoning": "Explanation of your synthesis decisions...",
  "final_clusters": [
    {
      "cluster_id": 0,
      "label": "Theme Name",
      "description": "What this cluster covers...",
      "paper_dois": ["doi1", "doi2"],
      "key_papers": ["most_important_doi1", "most_important_doi2"],
      "sub_themes": ["sub1", "sub2"],
      "conflicts": ["papers X and Y disagree on..."],
      "gaps": ["under-researched area..."],
      "source": "merged" | "bertopic" | "llm"
    }
  ]
}"""

OPUS_SYNTHESIS_USER_TEMPLATE = """I have {paper_count} papers to organize. Here are the two clustering analyses:

## Statistical Clustering (BERTopic)
{bertopic_summary}

## Semantic Clustering (LLM Analysis)
{llm_schema_summary}

## Paper Summaries (for reference)
{paper_summaries}

Synthesize these into a final thematic organization."""

CLUSTER_ANALYSIS_SYSTEM_PROMPT = """You are a research analyst providing deep analysis of a thematic cluster of papers.

For this cluster, produce:
1. **Narrative Summary**: A 2-3 paragraph overview of the theme, suitable as the introduction to a literature review section
2. **Timeline**: Key developments in chronological order
3. **Key Debates**: Main disagreements, tensions, or alternative positions
4. **Methodologies**: Common methodological approaches used
5. **Outstanding Questions**: Open research questions and gaps

Be specific and cite papers by their titles when making claims.

Output a JSON object:
{
  "narrative_summary": "...",
  "timeline": ["1995: Paper X established...", "2010: Paper Y challenged..."],
  "key_debates": ["Whether approach A or B is superior...", "The role of factor C..."],
  "methodologies": ["Survey studies (Paper A, Paper B)", "Experiments (Paper C)"],
  "outstanding_questions": ["How does X affect Y?", "Can findings generalize to..."]
}"""

CLUSTER_ANALYSIS_USER_TEMPLATE = """Analyze this thematic cluster:

Theme: {theme_name}
Description: {theme_description}

Papers in this cluster:
{papers_detail}

Provide a deep analysis of this research theme."""
