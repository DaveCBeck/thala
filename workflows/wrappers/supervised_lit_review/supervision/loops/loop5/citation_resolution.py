"""LLM-based resolution of invalid citations."""

import logging
import re

from pydantic import BaseModel, Field

from workflows.shared.llm_utils import get_structured_output, ModelTier
from ...store_query import SupervisionStoreQuery
from ...tools import create_paper_tools

logger = logging.getLogger(__name__)


class CitationFix(BaseModel):
    """A fix for an invalid citation."""

    original_text: str = Field(
        description="The exact text containing the invalid citation (50-150 chars for uniqueness)"
    )
    replacement_text: str = Field(
        description="The corrected text with valid citation, no citation, or rewritten claim"
    )
    reasoning: str = Field(description="Brief explanation of the fix")


class CitationResolutionResult(BaseModel):
    """Result of resolving invalid citations."""

    fixes: list[CitationFix] = Field(
        default_factory=list, description="List of fixes to apply"
    )


CITATION_RESOLUTION_SYSTEM = """You are fixing invalid citations in an academic literature review.

Each citation key you're given does NOT exist in Zotero - it's invalid and must be fixed.

## Available Tools

1. **search_papers(query, limit)** - Search the paper corpus
   - Find papers that could replace the invalid citation
   - Returns zotero_key, title, year, authors, relevance

2. **get_paper_content(zotero_key, max_chars)** - Get paper details
   - Verify a paper supports the claim before citing it

## For Each Invalid Citation

1. Find the sentence/claim containing the invalid [@KEY]
2. Decide how to fix it:
   - **Replace**: Find a valid paper that supports the claim using tools
   - **Remove**: If the citation is unnecessary (claim is common knowledge)
   - **Rewrite**: If no supporting paper exists, rewrite to remove the unsupported claim

## Output Format

For each fix, provide:
- `original_text`: The exact text containing [@INVALID_KEY] (50-150 chars for uniqueness)
- `replacement_text`: The fixed text (with valid [@KEY], no citation, or rewritten)
- `reasoning`: Why you made this choice

## Rules

- NEVER leave an invalid citation in place
- NEVER add a TODO marker - fix it now
- If you can't find a supporting paper, rewrite to remove the claim
- Use [@KEY] format for all citations where KEY is the zotero_key from search results"""


CITATION_RESOLUTION_USER = """Fix these invalid citations in the document.

## Invalid Citation Keys (do NOT exist in Zotero)
{invalid_keys}

## Document Excerpt
{document_excerpt}

## Topic
{topic}

Find and fix each invalid citation. Search for replacement papers or rewrite as needed."""


async def resolve_invalid_citations(
    document: str,
    invalid_keys: set[str],
    topic: str,
) -> str:
    """Use LLM to resolve invalid citations.

    Args:
        document: Full document text
        invalid_keys: Set of citation keys that don't exist in Zotero
        topic: Document topic for context

    Returns:
        Updated document with invalid citations fixed
    """
    if not invalid_keys:
        return document

    logger.info(f"Resolving {len(invalid_keys)} invalid citations")

    # Extract relevant excerpts containing invalid citations
    excerpts = []
    for key in invalid_keys:
        pattern = rf".{{0,200}}\[@{re.escape(key)}\].{{0,200}}"
        matches = re.findall(pattern, document, re.DOTALL)
        excerpts.extend(matches)

    if not excerpts:
        logger.warning("Could not find invalid citations in document")
        return document

    document_excerpt = "\n\n---\n\n".join(excerpts[:10])  # Limit to 10 excerpts

    # Set up tools
    store_query = SupervisionStoreQuery()
    paper_tools = create_paper_tools(store_query)

    try:
        user_prompt = CITATION_RESOLUTION_USER.format(
            invalid_keys=", ".join(f"[@{k}]" for k in sorted(invalid_keys)),
            document_excerpt=document_excerpt,
            topic=topic,
        )

        result = await get_structured_output(
            output_schema=CitationResolutionResult,
            user_prompt=user_prompt,
            system_prompt=CITATION_RESOLUTION_SYSTEM,
            tools=paper_tools,
            tier=ModelTier.SONNET,
            max_tokens=4096,
        )

        # Apply fixes
        updated = document
        fixes_applied = 0
        for fix in result.fixes:
            if fix.original_text in updated:
                updated = updated.replace(fix.original_text, fix.replacement_text, 1)
                fixes_applied += 1
                logger.debug(f"Applied citation fix: {fix.reasoning[:50]}...")
            else:
                logger.warning(
                    f"Could not find text to fix: {fix.original_text[:50]}..."
                )

        logger.info(f"Applied {fixes_applied}/{len(result.fixes)} citation fixes")

        # Fallback: strip any remaining invalid citations that weren't fixed
        for key in invalid_keys:
            pattern = rf"\[@{re.escape(key)}\]"
            if re.search(pattern, updated):
                logger.warning(f"Stripping unfixed invalid citation [@{key}]")
                updated = re.sub(pattern, "", updated)

        return updated

    finally:
        await store_query.close()
