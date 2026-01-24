"""Paper fallback mechanism for handling acquisition/processing failures.

When a paper fails to acquire or process (invalid PDF, metadata mismatch, download failure),
the FallbackManager provides the next-best alternative from the fallback queue.
"""

import logging
from typing import Optional

from workflows.research.academic_lit_review.state import (
    FallbackCandidate,
    FallbackSubstitution,
    PaperMetadata,
)

logger = logging.getLogger(__name__)


class FallbackManager:
    """Manages fallback paper selection and substitution tracking.

    The fallback queue is pre-sorted by relevance score (overflow papers first,
    then near-threshold papers). When a paper fails, this manager provides
    the next-best alternative and tracks all substitutions.
    """

    def __init__(
        self,
        fallback_queue: list[FallbackCandidate],
        paper_corpus: dict[str, PaperMetadata],
    ):
        """Initialize the fallback manager.

        Args:
            fallback_queue: Pre-sorted list of fallback candidates
                (overflow papers first, then near-threshold)
            paper_corpus: Full paper corpus for metadata lookup
        """
        self.queue = list(fallback_queue)  # Make a mutable copy
        self.corpus = paper_corpus
        self.used: set[str] = set()
        self.substitutions: list[FallbackSubstitution] = []
        self.exhausted_warnings: list[str] = []

    def get_fallback_for(
        self,
        failed_doi: str,
        failure_reason: str,
        failure_stage: str,
    ) -> Optional[PaperMetadata]:
        """Get the next-best fallback paper for a failed DOI.

        Args:
            failed_doi: DOI of the paper that failed
            failure_reason: Why it failed ("pdf_invalid", "metadata_mismatch", "acquisition_failed")
            failure_stage: Where it failed ("acquisition", "marker", "validation")

        Returns:
            PaperMetadata for the fallback paper, or None if no fallbacks available
        """
        for candidate in self.queue:
            candidate_doi = candidate.get("doi", "")
            if not candidate_doi or candidate_doi in self.used:
                continue

            # Mark as used
            self.used.add(candidate_doi)

            # Get full metadata from corpus
            fallback_metadata = self.corpus.get(candidate_doi)
            if not fallback_metadata:
                logger.warning(
                    f"Fallback candidate {candidate_doi} not found in corpus, skipping"
                )
                continue

            # Record substitution
            self.substitutions.append(
                FallbackSubstitution(
                    failed_doi=failed_doi,
                    fallback_doi=candidate_doi,
                    failure_reason=failure_reason,
                    failure_stage=failure_stage,
                )
            )

            logger.info(
                f"Fallback substitution: {failed_doi} -> {candidate_doi} "
                f"(score={candidate.get('relevance_score', 0):.2f}, "
                f"source={candidate.get('source', 'unknown')}, "
                f"reason={failure_reason})"
            )

            return fallback_metadata

        # No fallback available
        logger.warning(f"No fallback candidates available for failed paper: {failed_doi}")
        self.exhausted_warnings.append(failed_doi)
        return None

    def get_substitutions(self) -> list[FallbackSubstitution]:
        """Get all substitutions made so far."""
        return self.substitutions

    def get_exhausted_warnings(self) -> list[str]:
        """Get DOIs that failed with no fallback available."""
        return self.exhausted_warnings

    def get_remaining_queue(self) -> list[FallbackCandidate]:
        """Get remaining unused fallback candidates."""
        return [c for c in self.queue if c.get("doi", "") not in self.used]

    def get_stats(self) -> dict:
        """Get statistics about fallback usage."""
        return {
            "total_candidates": len(self.queue),
            "used_count": len(self.used),
            "remaining_count": len(self.queue) - len(self.used),
            "substitutions_made": len(self.substitutions),
            "exhausted_count": len(self.exhausted_warnings),
        }
