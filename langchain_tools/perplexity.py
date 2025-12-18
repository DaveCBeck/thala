"""
Perplexity search tools for LangChain.

Provides: perplexity_search, check_fact
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client Management (lazy singleton)
# ---------------------------------------------------------------------------

_perplexity_client = None


def _get_perplexity():
    """Get Perplexity httpx client (lazy init)."""
    global _perplexity_client
    if _perplexity_client is None:
        import httpx

        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError(
                "PERPLEXITY_API_KEY environment variable is required. "
                "Get one at https://perplexity.ai"
            )
        _perplexity_client = httpx.AsyncClient(
            base_url="https://api.perplexity.ai",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    return _perplexity_client


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------


class PerplexitySearchResult(BaseModel):
    """Individual Perplexity search result."""

    title: str
    url: str
    snippet: Optional[str] = None
    date: Optional[str] = None


class PerplexitySearchOutput(BaseModel):
    """Output schema for perplexity_search tool."""

    query: str
    total_results: int
    results: list[PerplexitySearchResult]


class FactCheckOutput(BaseModel):
    """Output schema for check_fact tool."""

    claim: str
    verdict: str  # "supported", "refuted", "partially_supported", "unverifiable"
    confidence: float  # 0.0 - 1.0
    explanation: str
    sources: list[PerplexitySearchResult]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
async def perplexity_search(
    query: str,
    limit: int = 5,
    domain_filter: Optional[list[str]] = None,
) -> dict:
    """Search the web using Perplexity AI.

    Perplexity provides AI-powered search with synthesized answers and citations.
    Use this for general web search alongside other search tools.

    Args:
        query: What to search for
        limit: Maximum number of results (default 5, max 20)
        domain_filter: Optional list of domains to restrict search to

    Returns:
        Search results with titles, URLs, and snippets.
    """
    client = _get_perplexity()
    limit = min(max(1, limit), 20)

    try:
        payload = {
            "query": query,
            "max_results": limit,
        }
        if domain_filter:
            payload["search_domain_filter"] = domain_filter

        response = await client.post("/search", json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", [])[:limit]:
            results.append(
                PerplexitySearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet"),
                    date=item.get("date"),
                )
            )

        output = PerplexitySearchOutput(
            query=query,
            total_results=len(results),
            results=results,
        )
        logger.debug(f"perplexity_search returned {len(results)} results for: {query}")
        return output.model_dump(mode="json")

    except Exception as e:
        logger.error(f"perplexity_search failed: {e}")
        return PerplexitySearchOutput(
            query=query,
            total_results=0,
            results=[],
        ).model_dump(mode="json")


@tool
async def check_fact(
    claim: str,
    context: Optional[str] = None,
) -> dict:
    """Verify a factual claim using Perplexity AI.

    Use this to fact-check specific claims or statements during research.
    Returns a verdict with supporting evidence.

    Args:
        claim: The factual claim to verify
        context: Optional context about where the claim came from

    Returns:
        Verification result with verdict, confidence, and sources.
    """
    client = _get_perplexity()

    try:
        # Construct a fact-checking query
        fact_check_query = f"Is this claim accurate? Provide evidence: '{claim}'"
        if context:
            fact_check_query += f" Context: {context}"

        payload = {
            "query": fact_check_query,
            "max_results": 10,  # Get more sources for fact-checking
        }

        response = await client.post("/search", json=payload)
        response.raise_for_status()
        data = response.json()

        # Parse results
        results = []
        for item in data.get("results", []):
            results.append(
                PerplexitySearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet"),
                    date=item.get("date"),
                )
            )

        # Use LLM to synthesize verdict from results
        from workflows.shared.llm_utils import ModelTier, get_llm

        llm = get_llm(ModelTier.HAIKU)

        # Build evidence summary from snippets
        evidence_summary = "\n".join(
            f"- {r.title}: {r.snippet[:300]}..." if r.snippet and len(r.snippet) > 300 else f"- {r.title}: {r.snippet or 'No snippet'}"
            for r in results[:5]
        )

        synthesis_prompt = f"""Analyze these search results to fact-check the claim.

Claim: {claim}

Evidence from search:
{evidence_summary}

Respond with ONLY valid JSON (no markdown):
{{
  "verdict": "supported" or "refuted" or "partially_supported" or "unverifiable",
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation of the verdict based on evidence"
}}"""

        llm_response = await llm.ainvoke([{"role": "user", "content": synthesis_prompt}])
        content = llm_response.content.strip()

        # Parse JSON from response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        verdict_data = json.loads(content)

        output = FactCheckOutput(
            claim=claim,
            verdict=verdict_data.get("verdict", "unverifiable"),
            confidence=float(verdict_data.get("confidence", 0.5)),
            explanation=verdict_data.get("explanation", ""),
            sources=results[:5],
        )

        logger.info(
            f"check_fact: '{claim[:50]}...' -> {output.verdict} "
            f"(conf: {output.confidence:.2f})"
        )
        return output.model_dump(mode="json")

    except Exception as e:
        logger.error(f"check_fact failed: {e}")
        return FactCheckOutput(
            claim=claim,
            verdict="unverifiable",
            confidence=0.0,
            explanation=f"Fact-check failed: {e}",
            sources=[],
        ).model_dump(mode="json")
