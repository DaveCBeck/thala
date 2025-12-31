"""Shared utilities for academic literature review workflow.

Contains:
- OpenAlex result conversion to PaperMetadata
- Relevance scoring prompts and functions
- Deduplication helpers
"""

import logging
from datetime import datetime
from typing import Optional

from langchain_tools.openalex import OpenAlexWork
from workflows.research.subgraphs.academic_lit_review.state import (
    PaperAuthor,
    PaperMetadata,
)
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    invoke_with_cache,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Conversion Utilities
# =============================================================================


def convert_to_paper_metadata(
    work: OpenAlexWork | dict,
    discovery_stage: int = 0,
    discovery_method: str = "keyword",
) -> Optional[PaperMetadata]:
    """Convert an OpenAlex work to PaperMetadata.

    Args:
        work: OpenAlex work object or dict
        discovery_stage: Which diffusion stage discovered this paper
        discovery_method: How this paper was discovered

    Returns:
        PaperMetadata or None if missing required fields (DOI)
    """
    # Handle both Pydantic model and dict
    if hasattr(work, "model_dump"):
        work_dict = work.model_dump()
    elif hasattr(work, "dict"):
        work_dict = work.dict()
    else:
        work_dict = dict(work)

    doi = work_dict.get("doi")
    if not doi:
        logger.debug(f"Skipping work without DOI: {work_dict.get('title', 'Unknown')}")
        return None

    # Normalize DOI (remove https://doi.org/ prefix)
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    # Parse authors
    authors = []
    for author_data in work_dict.get("authors", []):
        if isinstance(author_data, dict):
            authors.append(
                PaperAuthor(
                    name=author_data.get("name", "Unknown"),
                    author_id=author_data.get("author_id"),
                    institution=author_data.get("institution"),
                    orcid=author_data.get("orcid"),
                )
            )

    # Extract year from publication_date
    pub_date = work_dict.get("publication_date", "")
    year = 0
    if pub_date and len(pub_date) >= 4:
        try:
            year = int(pub_date[:4])
        except ValueError:
            pass

    # Extract OpenAlex ID from URL if present
    openalex_id = ""
    work_id = work_dict.get("id", "") or work_dict.get("openalex_id", "")
    if work_id:
        openalex_id = work_id.split("/")[-1] if "/" in work_id else work_id

    return PaperMetadata(
        doi=doi_clean,
        title=work_dict.get("title", "Untitled"),
        authors=authors,
        publication_date=pub_date,
        year=year,
        venue=work_dict.get("source_name"),
        cited_by_count=work_dict.get("cited_by_count", 0),
        abstract=work_dict.get("abstract"),
        openalex_id=openalex_id,
        primary_topic=work_dict.get("primary_topic"),
        is_oa=work_dict.get("is_oa", False),
        oa_url=work_dict.get("oa_url"),
        oa_status=work_dict.get("oa_status"),
        referenced_works=work_dict.get("referenced_works", []),
        citing_works_count=work_dict.get("cited_by_count", 0),
        retrieved_at=datetime.utcnow(),
        discovery_stage=discovery_stage,
        discovery_method=discovery_method,
    )


def deduplicate_papers(
    papers: list[PaperMetadata],
    existing_dois: set[str] | None = None,
) -> list[PaperMetadata]:
    """Deduplicate papers by DOI, keeping the first occurrence.

    Args:
        papers: List of PaperMetadata to deduplicate
        existing_dois: Optional set of DOIs already in corpus to exclude

    Returns:
        Deduplicated list of papers
    """
    existing_dois = existing_dois or set()
    seen_dois = set()
    unique_papers = []

    for paper in papers:
        doi = paper.get("doi")
        if doi and doi not in seen_dois and doi not in existing_dois:
            seen_dois.add(doi)
            unique_papers.append(paper)

    logger.debug(
        f"Deduplicated {len(papers)} papers to {len(unique_papers)} "
        f"(excluded {len(papers) - len(unique_papers)} duplicates)"
    )
    return unique_papers


# =============================================================================
# Relevance Scoring
# =============================================================================

RELEVANCE_SCORING_SYSTEM = """You are an academic literature review assistant evaluating paper relevance.

Given a research topic and a paper's metadata, score its relevance from 0.0 to 1.0:
- 1.0: Directly addresses the core topic, essential reading
- 0.8-0.9: Highly relevant, addresses key aspects
- 0.6-0.7: Moderately relevant, provides useful context
- 0.4-0.5: Tangentially related, may have some value
- 0.2-0.3: Loosely related, minimal direct relevance
- 0.0-0.1: Not relevant to the topic

Consider:
- Title and abstract alignment with topic
- Methodology relevance (if applicable)
- Theoretical framework fit
- Disciplinary alignment

Output ONLY a JSON object with:
{
  "relevance_score": <float 0.0-1.0>,
  "reasoning": "<brief 1-2 sentence explanation>"
}"""

RELEVANCE_SCORING_USER_TEMPLATE = """Research Topic: {topic}
Research Questions: {research_questions}

Paper to Evaluate:
- Title: {title}
- Authors: {authors}
- Year: {year}
- Venue: {venue}
- Abstract: {abstract}
- Primary Topic: {primary_topic}

Evaluate this paper's relevance to the research topic."""


async def score_paper_relevance(
    paper: PaperMetadata,
    topic: str,
    research_questions: list[str],
    tier: ModelTier = ModelTier.HAIKU,
) -> tuple[float, str]:
    """Score a single paper's relevance to the research topic.

    Args:
        paper: Paper metadata to evaluate
        topic: Research topic
        research_questions: List of research questions
        tier: Model tier for scoring

    Returns:
        Tuple of (relevance_score, reasoning)
    """
    import json

    llm = get_llm(tier=tier)

    # Format authors
    authors_str = ", ".join(
        a.get("name", "") for a in paper.get("authors", [])[:5]
    )
    if len(paper.get("authors", [])) > 5:
        authors_str += " et al."

    user_prompt = RELEVANCE_SCORING_USER_TEMPLATE.format(
        topic=topic,
        research_questions="; ".join(research_questions[:3]),
        title=paper.get("title", "Unknown"),
        authors=authors_str or "Unknown",
        year=paper.get("year", "Unknown"),
        venue=paper.get("venue", "Unknown"),
        abstract=(paper.get("abstract") or "No abstract available")[:1000],
        primary_topic=paper.get("primary_topic", "Not specified"),
    )

    try:
        response = await invoke_with_cache(
            llm,
            system_prompt=RELEVANCE_SCORING_SYSTEM,
            user_prompt=user_prompt,
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)
        score = float(result.get("relevance_score", 0.5))
        reasoning = result.get("reasoning", "")

        # Clamp score to valid range
        score = max(0.0, min(1.0, score))

        return score, reasoning

    except Exception as e:
        logger.warning(f"Failed to score relevance for {paper.get('title', 'Unknown')}: {e}")
        # Default to moderate relevance on failure
        return 0.5, f"Scoring failed: {e}"


async def batch_score_relevance(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float = 0.6,
    tier: ModelTier = ModelTier.HAIKU,
    max_concurrent: int = 10,
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score multiple papers' relevance and filter by threshold.

    Args:
        papers: Papers to evaluate
        topic: Research topic
        research_questions: List of research questions
        threshold: Minimum relevance score to include
        tier: Model tier for scoring
        max_concurrent: Maximum concurrent scoring calls

    Returns:
        Tuple of (relevant_papers, rejected_papers)
    """
    import asyncio

    if not papers:
        return [], []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_limit(paper: PaperMetadata) -> tuple[PaperMetadata, float, str]:
        async with semaphore:
            score, reasoning = await score_paper_relevance(
                paper, topic, research_questions, tier
            )
            return paper, score, reasoning

    # Score all papers concurrently
    tasks = [score_with_limit(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    relevant = []
    rejected = []

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Relevance scoring task failed: {result}")
            continue

        paper, score, reasoning = result
        # Attach score to paper for downstream filtering
        paper["relevance_score"] = score
        if score >= threshold:
            relevant.append(paper)
            logger.debug(
                f"RELEVANT ({score:.2f}): {paper.get('title', 'Unknown')[:50]} - {reasoning}"
            )
        else:
            rejected.append(paper)
            logger.debug(
                f"REJECTED ({score:.2f}): {paper.get('title', 'Unknown')[:50]} - {reasoning}"
            )

    logger.info(
        f"Relevance filtering: {len(relevant)} relevant, {len(rejected)} rejected "
        f"(threshold={threshold})"
    )

    return relevant, rejected


# =============================================================================
# Query Generation Prompts
# =============================================================================

GENERATE_ACADEMIC_SEARCH_QUERIES_SYSTEM = """You are an expert academic researcher generating search queries for OpenAlex.

Generate 3-5 search queries optimized for finding relevant academic literature on the given topic.

Query strategies to use:
1. Core topic + methodology terms (e.g., "deep learning image classification survey")
2. Broader field + specific concepts (e.g., "computer vision neural networks review")
3. Related terminology / synonyms (e.g., "convolutional networks visual recognition")
4. Historical framing for seminal works (e.g., "artificial neural networks early foundations")
5. Recent framing for current research (e.g., "transformer architecture vision 2023")

Guidelines:
- Use academic/scholarly terminology
- Include methodological keywords when relevant
- Mix broad and specific queries
- Consider field-specific vocabulary
- Aim for queries that would find peer-reviewed literature

Output a JSON object:
{
  "queries": ["query1", "query2", "query3", ...]
}"""

GENERATE_ACADEMIC_SEARCH_QUERIES_USER = """Topic: {topic}

Research Questions:
{research_questions}

Focus Areas: {focus_areas}

Generate academic search queries to find relevant literature on this topic."""


async def generate_search_queries(
    topic: str,
    research_questions: list[str],
    focus_areas: list[str] | None = None,
    tier: ModelTier = ModelTier.HAIKU,
) -> list[str]:
    """Generate academic search queries for the given topic.

    Args:
        topic: Research topic
        research_questions: List of research questions
        focus_areas: Optional specific areas to focus on
        tier: Model tier for generation

    Returns:
        List of search queries
    """
    import json

    llm = get_llm(tier=tier)

    user_prompt = GENERATE_ACADEMIC_SEARCH_QUERIES_USER.format(
        topic=topic,
        research_questions="\n".join(f"- {q}" for q in research_questions),
        focus_areas=", ".join(focus_areas) if focus_areas else "None specified",
    )

    try:
        response = await invoke_with_cache(
            llm,
            system_prompt=GENERATE_ACADEMIC_SEARCH_QUERIES_SYSTEM,
            user_prompt=user_prompt,
        )

        content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
        content = content.strip()

        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        result = json.loads(content)
        queries = result.get("queries", [])

        if not queries:
            # Fallback to topic as query
            queries = [topic]

        logger.info(f"Generated {len(queries)} search queries for topic: {topic[:50]}...")
        return queries

    except Exception as e:
        logger.error(f"Failed to generate search queries: {e}")
        # Fallback to basic queries
        return [topic, f"{topic} review", f"{topic} survey"]
