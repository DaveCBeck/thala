"""Pydantic schemas for structured LLM outputs in fact-check workflow."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Pre-screening Schemas
# =============================================================================


class FactCheckScreeningResult(BaseModel):
    """Result of screening sections for fact-checking.

    Simplified schema - just return the section ID lists, no detailed per-section analysis.
    This keeps output small to avoid token limits with Haiku.
    """

    sections_to_check: list[str] = Field(
        default_factory=list,
        description="Section IDs containing factual claims that need verification",
    )
    sections_to_skip: list[str] = Field(
        default_factory=list,
        description="Section IDs that are methodological/narrative/config - skip these",
    )
    screening_summary: str = Field(
        default="Screening incomplete",
        description="Brief summary: X factual, Y methodological, Z narrative/config",
    )


# =============================================================================
# Fact-check Schemas
# =============================================================================


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim_text: str = Field(description="The claim being verified")
    claim_type: Literal[
        "factual",  # Verifiable fact
        "methodological",  # Research methodology choice
        "interpretive",  # Interpretation of evidence
        "established",  # Well-established knowledge
    ] = Field(description="Type of claim")

    verdict: Literal[
        "supported",  # Evidence supports the claim
        "contradicted",  # Evidence contradicts the claim
        "partially_supported",  # Mixed evidence
        "unverifiable",  # Cannot verify from available sources
        "not_applicable",  # Not a verifiable claim
    ] = Field(description="Verification verdict")

    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(description="Summary of evidence found")
    source_used: Literal["corpus", "perplexity", "both", "none"] = Field(
        description="What source provided the verification"
    )


class FactCheckResult(BaseModel):
    """Result of fact-checking a section."""

    section_id: str = Field(description="ID of the checked section")
    claims_checked: int = Field(description="Number of claims examined")
    claims_verified: list[ClaimVerification] = Field(default_factory=list)

    suggested_edits: list["VerifiedEdit"] = Field(
        default_factory=list,
        description="Corrections suggested based on fact-checking",
    )
    unresolved_issues: list[str] = Field(
        default_factory=list,
        description="Issues that could not be resolved (logged at INFO)",
    )


# =============================================================================
# Reference-check Schemas
# =============================================================================


class CitationValidation(BaseModel):
    """Validation result for a single citation."""

    citation_key: str = Field(description="The citation key [@KEY]")
    exists_in_zotero: bool = Field(description="Whether key exists in Zotero")
    exists_in_corpus: bool = Field(description="Whether paper content is in corpus")
    supports_claim: Optional[bool] = Field(
        default=None,
        description="Whether paper content supports the claim it's cited for",
    )
    validation_notes: str = Field(default="")


class ReferenceCheckResult(BaseModel):
    """Result of checking citations in a section."""

    section_id: str = Field(description="ID of the checked section")
    citations_found: list[str] = Field(description="All citation keys in section")
    validations: list[CitationValidation] = Field(default_factory=list)

    invalid_citations: list[str] = Field(
        default_factory=list,
        description="Citations that don't exist or are invalid",
    )
    unsupported_citations: list[str] = Field(
        default_factory=list,
        description="Citations that don't support their claimed context",
    )

    suggested_edits: list["VerifiedEdit"] = Field(
        default_factory=list,
        description="Corrections for citation issues",
    )


# =============================================================================
# Edit Schemas
# =============================================================================


class VerifiedEdit(BaseModel):
    """An edit that has been validated for application."""

    find: str = Field(
        min_length=10,
        max_length=500,
        description="Text to find (must be unique in document)",
    )
    replace: str = Field(description="Replacement text")
    position_hint: str = Field(
        description="Location hint (e.g., 'in section X' or 'after paragraph starting with Y')"
    )

    edit_type: Literal[
        "fact_correction",  # Correct a factual error
        "citation_fix",  # Fix an invalid citation
        "citation_add",  # Add a missing citation
        "clarity",  # Clarify ambiguous claim
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    source_reference: Optional[str] = Field(
        default=None,
        description="Citation key or URL supporting this edit",
    )
    justification: str = Field(description="Why this edit is needed")


# Update forward references
FactCheckResult.model_rebuild()
ReferenceCheckResult.model_rebuild()
