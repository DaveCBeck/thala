"""Prompts for Haiku relevance checking to determine if a language has meaningful content."""

RELEVANCE_CHECK_SYSTEM = """You are assessing whether a research topic has meaningful discussion in a specific language.

Your task: Determine if searching in {language_name} will yield valuable, unique content for the research topic.

Consider:
1. Is this topic discussed in {language_name} sources?
2. Would {language_name} sources offer unique perspectives not found in English?
3. Are there regional/cultural aspects relevant to {language_name} speakers?
4. Is there academic, journalistic, or professional coverage in this language?

Be conservative - only say "yes" if you're reasonably confident valuable content exists.
Be decisive - if the quick search results show relevant discussions, mark as meaningful.

Respond with structured JSON matching the schema provided."""

RELEVANCE_CHECK_USER = """Topic: {topic}

Research questions:
{research_questions}

Quick search results from {language_name}:
{quick_search_results}

Based on these preliminary results, does meaningful discussion exist in {language_name} for this topic?"""
