"""Query generation prompts for different researcher types."""

GENERATE_WEB_QUERIES_SYSTEM = """Generate 3-4 web search queries for general search engines.

Focus on finding recent, authoritative web sources:
- Official websites, documentation, and company pages
- News articles, journalism, and industry publications
- Blog posts from recognized experts and practitioners
- Forums, discussions, and community resources (Reddit, HN, Stack Overflow)
- Product pages, comparisons, and reviews
- Review articles and accessible summaries of scientific developments
- Government agency pages, regulatory documents, and program announcements

Use natural language queries that work well with Google/Bing.
Include year references (2025, 2026) for current topics.
Make queries SPECIFIC to the research question — use domain-specific terminology.
Focus only on the research topic - do not include any system metadata."""

GENERATE_ACADEMIC_QUERIES_SYSTEM = """Generate 2-3 search queries optimized for academic literature databases.

OpenAlex searches peer-reviewed research across ALL disciplines. Optimize your queries:

**For STEM topics:**
- Include methodology terms: "meta-analysis", "RCT", "longitudinal study", "cohort study"
- Use academic phrasing: "effects of X on Y", "relationship between X and Y"

**For humanities & social sciences:**
- Literature/Language: "literary analysis", "narrative theory", "discourse analysis", "semiotics"
- Arts: "aesthetic theory", "art criticism", "musicology", "film studies"
- Social Science: "qualitative study", "ethnography", "case study", "critical theory"
- Philosophy: "phenomenology", "hermeneutics", "epistemology"

**General guidance:**
- Use domain-specific terminology from academic papers
- Avoid colloquial language, product names, and current events
- Focus on concepts and theories rather than specific tools

Examples:
- "meditation neural plasticity fMRI meta-analysis" (neuroscience)
- "postcolonial literature narrative identity" (literary studies)
- "social media political polarization qualitative" (social science)
- "jazz improvisation cognitive processes" (musicology)

Make queries likely to match academic paper titles and abstracts.
Focus only on the research topic - do not include any system metadata."""

GENERATE_BOOK_QUERIES_SYSTEM = """Generate 2-3 search queries optimized for book databases.

Books excel for foundational knowledge, theory, and comprehensive treatments.

**Best suited for:**
- Foundational theory and classic works in any field
- Comprehensive overviews and textbooks
- Historical context and development of ideas
- Philosophy, literary criticism, art history
- Practical guides and "how-to" from experts
- Biographies of influential figures

**For humanities & arts topics:**
- Literature: author studies, genre analysis, literary movements
- Arts: art history, music theory, film criticism, aesthetics
- Language: linguistics, translation theory, language learning
- Philosophy: major philosophers, schools of thought, applied ethics

**Query strategies:**
- Use broad topic terms (books cover topics comprehensively)
- Include author names if you know experts in the field
- Include "introduction to", "handbook", "companion to" for overviews
- Use established terminology and movement names

Examples:
- "cognitive psychology attention" (psychology)
- "postmodern fiction theory" (literary studies)
- "Renaissance art history" (art history)
- "linguistics syntax semantics introduction" (language)

Avoid current events and version-specific technical details.
Focus only on the research topic - do not include any system metadata."""
