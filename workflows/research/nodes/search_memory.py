"""
Search memory node.

Searches Thala's memory stores (top_of_mind, coherence, store) for relevant
existing knowledge before conducting web research.
"""

import logging
from typing import Any

from langchain_tools.search_memory import search_memory
from workflows.research.state import DeepResearchState
from workflows.shared.llm_utils import ModelTier, summarize_text

logger = logging.getLogger(__name__)

# Max results per query
MAX_RESULTS_PER_QUERY = 5
# Max total results
MAX_TOTAL_RESULTS = 15


async def search_memory_node(state: DeepResearchState) -> dict[str, Any]:
    """Search Thala's memory stores for relevant existing knowledge.

    Searches: top_of_mind, coherence, store

    Returns:
        - memory_findings: list of search results
        - memory_context: summarized context string
        - current_status: updated status
    """
    brief = state.get("research_brief")
    if not brief:
        logger.warning("No research brief available for memory search")
        return {
            "memory_findings": [],
            "memory_context": "No research brief available.",
            "current_status": "iterating_plan",
        }

    # Build search queries from brief
    queries = [brief["topic"]]
    queries.extend(brief.get("key_questions", [])[:3])

    all_results = []

    for query in queries:
        try:
            result = await search_memory.ainvoke({
                "query": query,
                "limit": MAX_RESULTS_PER_QUERY,
                "stores": ["top_of_mind", "coherence", "store"],
            })

            results_list = result.get("results", [])
            all_results.extend(results_list)

            logger.debug(f"Memory search for '{query[:30]}...': {len(results_list)} results")

        except Exception as e:
            logger.warning(f"Memory search failed for '{query[:30]}...': {e}")

    # Deduplicate by ID
    seen_ids = set()
    unique_results = []
    for r in all_results:
        record_id = r.get("id", "")
        if record_id and record_id not in seen_ids:
            seen_ids.add(record_id)
            unique_results.append(r)

    # Limit total results
    unique_results = unique_results[:MAX_TOTAL_RESULTS]

    # Summarize memory findings
    if unique_results:
        # Include score info for debugging
        findings_text = "\n\n".join([
            f"[{r.get('source_store', 'unknown')}] (score: {r.get('score', 'N/A')})\n{r.get('content', '')[:500]}"
            for r in unique_results
        ])

        try:
            memory_context = await summarize_text(
                text=findings_text,
                target_words=300,
                context=f"Summarize what the user already knows about: {brief['topic']}. If none of the content is relevant to this topic, say so clearly.",
                tier=ModelTier.HAIKU,
            )
        except Exception as e:
            logger.warning(f"Failed to summarize memory context: {e}")
            memory_context = findings_text[:1000]

        logger.info(f"Found {len(unique_results)} relevant memory records")
    else:
        memory_context = "No relevant prior knowledge found on this topic."
        logger.info("No memory records passed relevance thresholds")

    # Update the research brief with memory context
    updated_brief = dict(brief)
    updated_brief["memory_context"] = memory_context

    return {
        "memory_findings": unique_results,
        "memory_context": memory_context,
        "research_brief": updated_brief,
        "current_status": "iterating_plan",
    }
