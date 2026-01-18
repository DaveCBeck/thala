"""LLM prompts for extraction."""

L0_SIZE_THRESHOLD_FOR_L2 = 150_000


def format_paper_extraction_system(topic: str, research_questions: list[str]) -> str:
    """Format the paper extraction system prompt with research context."""
    questions_formatted = "\n".join(f"- {q}" for q in research_questions)

    return f"""Analyze this academic paper and extract structured information relevant to the following research:

Topic: {topic}

Research Questions:
{questions_formatted}

Extract the following in JSON format:
{{
  "key_findings": ["3-5 findings from the paper relevant to the research questions"],
  "methodology": "Brief research method description (1-2 sentences)",
  "limitations": ["Stated limitations relevant to our research"],
  "future_work": ["Suggested future directions relevant to our topic"],
  "themes": ["3-5 topic tags for clustering"]
}}

Be specific and grounded in the paper content. Do not hallucinate information.
If a field is not present in the paper or not relevant to our research questions, use an empty list or brief note."""


def format_metadata_extraction_system(topic: str, research_questions: list[str]) -> str:
    """Format the metadata extraction system prompt with research context."""
    questions_formatted = "\n".join(f"- {q}" for q in research_questions)

    return f"""Analyze this academic paper's metadata and abstract to extract structured information relevant to the following research:

Topic: {topic}

Research Questions:
{questions_formatted}

Extract the following in JSON format:
{{
  "key_findings": ["2-3 inferred findings based on the abstract, relevant to our research"],
  "methodology": "Inferred methodology from the abstract (1-2 sentences)",
  "limitations": [],
  "future_work": [],
  "themes": ["3-5 topic tags based on title and abstract"]
}}

Note: This is based only on metadata/abstract, not full text. Be conservative and extract only what is clearly stated and relevant to our research questions."""


# Legacy prompts for backward compatibility (used when no research context available)
PAPER_SUMMARY_EXTRACTION_SYSTEM = """Analyze this academic paper and extract structured information.

Extract the following in JSON format:
{
  "key_findings": ["3-5 specific findings from the paper"],
  "methodology": "Brief research method description (1-2 sentences)",
  "limitations": ["Stated limitations from the paper"],
  "future_work": ["Suggested future research directions"],
  "themes": ["3-5 topic tags for clustering"]
}

Be specific and grounded in the paper content. Do not hallucinate information.
If a field is not present in the paper, use an empty list or brief note."""

METADATA_SUMMARY_EXTRACTION_SYSTEM = """Analyze this academic paper's metadata and abstract to extract structured information.

Extract the following in JSON format:
{
  "key_findings": ["2-3 inferred findings based on the abstract"],
  "methodology": "Inferred methodology from the abstract (1-2 sentences)",
  "limitations": [],
  "future_work": [],
  "themes": ["3-5 topic tags based on title and abstract"]
}

Note: This is based only on metadata/abstract, not full text. Be conservative and extract only what is clearly stated."""
