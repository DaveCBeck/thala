"""Store query service for supervision loops.

Provides dynamic access to Elasticsearch store for detailed paper content
at different compression levels.
"""

import logging
import re
from typing import Any, Optional
from uuid import UUID

from langchain_tools.base import get_store_manager, StoreManager
from workflows.academic_lit_review.state import PaperSummary

logger = logging.getLogger(__name__)


def extract_citation_keys(text: str) -> list[str]:
    """Extract [@KEY] citation keys from text.

    Args:
        text: Document text containing citations

    Returns:
        List of citation keys (without brackets)
    """
    return re.findall(r"\[@([^\]]+)\]", text)


class SupervisionStoreQuery:
    """Query service for supervision loops to access detailed paper content.

    Enables loops to dynamically fetch paper content at different compression levels:
    - L0: Full original document (large)
    - L1: Short summary (~100 words)
    - L2: 10:1 compressed summary (medium)

    Note: es_record_id in paper_summaries is always the L0 UUID. For L1/L2,
    we use get_by_source_id() which searches by source_ids field.

    Usage:
        store_query = SupervisionStoreQuery(paper_summaries)
        content = await store_query.get_paper_content(doi, compression_level=2)
    """

    def __init__(self, paper_summaries: dict[str, PaperSummary]):
        """Initialize with paper summaries for DOI/key mapping.

        Args:
            paper_summaries: Dict of DOI -> PaperSummary from workflow state
        """
        self.paper_summaries = paper_summaries
        self._store_manager: Optional[StoreManager] = None
        self._key_to_doi = self._build_key_mapping()

    def _build_key_mapping(self) -> dict[str, str]:
        """Build zotero_key -> DOI mapping for citation resolution."""
        mapping = {}
        for doi, summary in self.paper_summaries.items():
            zotero_key = summary.get("zotero_key")
            if zotero_key:
                mapping[zotero_key] = doi
        return mapping

    @property
    def store_manager(self) -> StoreManager:
        """Get store manager (lazy initialization)."""
        if self._store_manager is None:
            self._store_manager = get_store_manager()
        return self._store_manager

    def resolve_citation_key(self, key: str) -> Optional[str]:
        """Resolve a citation key to its DOI.

        Args:
            key: Citation key from [@KEY] format

        Returns:
            DOI string if found, None otherwise
        """
        return self._key_to_doi.get(key)

    async def get_paper_content(
        self,
        doi: str,
        compression_level: int = 2,
    ) -> Optional[str]:
        """Fetch paper content from store at specified compression level.

        The es_record_id stored in paper_summaries is the L0 (original) UUID.
        For L1/L2 lookups, we use get_by_source_id() which finds compressed
        records by their source_ids field (which contains the L0 UUID).

        Args:
            doi: Paper DOI
            compression_level: 0=full, 1=short summary, 2=10:1 summary

        Returns:
            Content string or None if not found
        """
        summary = self.paper_summaries.get(doi)
        if not summary:
            logger.debug(f"No summary found for DOI: {doi}")
            return None

        es_record_id = summary.get("es_record_id")
        if not es_record_id:
            logger.debug(f"No es_record_id for DOI: {doi}, using summary content")
            return self._get_fallback_content(summary)

        try:
            record_uuid = UUID(es_record_id)
            store = self.store_manager.es_stores.store

            # For L0, use direct ID lookup
            # For L1/L2, use source_id lookup since es_record_id is the L0 UUID
            if compression_level == 0:
                record = await store.get(record_uuid, compression_level=0)
            else:
                record = await store.get_by_source_id(
                    record_uuid,
                    compression_level=compression_level,
                )

            if record:
                return record.content

            # Not found at specified level - try alternative
            logger.debug(
                f"No record at L{compression_level} for DOI: {doi} "
                f"(L0 id: {es_record_id}), trying fallback"
            )

            alt_level = 1 if compression_level == 2 else 2
            if alt_level != 0:
                record = await store.get_by_source_id(record_uuid, alt_level)
            else:
                record = await store.get(record_uuid, compression_level=0)

            if record:
                logger.info(f"Found record at L{alt_level} instead of L{compression_level}")
                return record.content

            return self._get_fallback_content(summary)

        except Exception as e:
            logger.warning(f"Error fetching content for {doi} (id: {es_record_id}): {e}")
            return self._get_fallback_content(summary)

    def _get_fallback_content(self, summary: dict) -> Optional[str]:
        """Get fallback content from summary when ES lookup fails.

        Constructs content from available summary fields.

        Args:
            summary: Paper summary dict

        Returns:
            Formatted content string or None
        """
        parts = []

        title = summary.get("title", "")
        if title:
            parts.append(f"# {title}")

        short_summary = summary.get("short_summary", "")
        if short_summary:
            parts.append(f"\n## Summary\n{short_summary}")

        methodology = summary.get("methodology", "")
        if methodology:
            parts.append(f"\n## Methodology\n{methodology}")

        key_findings = summary.get("key_findings", [])
        if key_findings:
            findings_text = "\n".join(f"- {f}" for f in key_findings)
            parts.append(f"\n## Key Findings\n{findings_text}")

        claims = summary.get("claims", [])
        if claims:
            claims_text = "\n".join(f"- {c}" for c in claims)
            parts.append(f"\n## Claims\n{claims_text}")

        if not parts:
            return None

        return "\n".join(parts)

    async def get_papers_for_section(
        self,
        section_content: str,
        max_papers: int = 10,
        compression_level: int = 2,
        max_total_chars: int = 50000,
    ) -> dict[str, str]:
        """Fetch detailed content for papers cited in a section.

        Extracts [@KEY] citations from section, maps to DOIs, fetches content.
        Respects total character budget to avoid context overflow.

        Args:
            section_content: Section text to extract citations from
            max_papers: Maximum papers to fetch
            compression_level: Which store level to query (0=full, 1=short, 2=10:1)
            max_total_chars: Total character budget across all papers

        Returns:
            Dict of DOI -> content for papers with available content
        """
        cited_keys = extract_citation_keys(section_content)
        logger.debug(f"Found {len(cited_keys)} citation keys in section")

        results: dict[str, str] = {}
        total_chars = 0
        fetched_count = 0

        for key in cited_keys:
            if fetched_count >= max_papers:
                break

            doi = self.resolve_citation_key(key)
            if not doi:
                continue

            # Skip if already fetched (duplicate citation)
            if doi in results:
                continue

            content = await self.get_paper_content(doi, compression_level)
            if content:
                # Check budget
                if total_chars + len(content) > max_total_chars:
                    logger.debug(
                        f"Character budget exceeded, stopping at {len(results)} papers"
                    )
                    break
                results[doi] = content
                total_chars += len(content)
                fetched_count += 1

        logger.info(
            f"Fetched {len(results)} papers ({total_chars} chars) at L{compression_level}"
        )
        return results

    def get_data_manifest(self) -> dict[str, dict[str, Any]]:
        """Return manifest of available data for each paper.

        Helps LLM understand what data is queryable and field sizes.

        Returns:
            Dict of DOI -> {title, zotero_key, has_es_record, field sizes...}
        """
        manifest: dict[str, dict[str, Any]] = {}
        for doi, summary in self.paper_summaries.items():
            manifest[doi] = {
                "title": summary.get("title", "")[:80],
                "zotero_key": summary.get("zotero_key", ""),
                "has_es_record": bool(summary.get("es_record_id")),
                "short_summary_chars": len(summary.get("short_summary", "")),
                "key_findings_count": len(summary.get("key_findings", [])),
                "methodology_chars": len(summary.get("methodology", "")),
                "claims_count": len(summary.get("claims", [])),
            }
        return manifest
