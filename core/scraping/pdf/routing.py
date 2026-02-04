"""Routing decisions based on document analysis.

This module separates routing logic from document analysis for:
- Testability (mock analysis, test routing independently)
- Configurability (routing thresholds can change without re-analyzing)
"""

from dataclasses import dataclass

from .analysis import DocumentAnalysis, DocumentComplexity


@dataclass
class RouteDecision:
    """Routing decision for document processing."""

    queue: str  # "cpu" or "gpu"
    confidence: float
    recommended_batch_size: int


def determine_route(analysis: DocumentAnalysis) -> RouteDecision:
    """Make routing decision based on document analysis.

    Args:
        analysis: DocumentAnalysis from analyze_document()

    Returns:
        RouteDecision with queue, confidence, and recommended batch size
    """
    # Scanned docs always need GPU (OCR required)
    if analysis.is_scanned:
        return RouteDecision(queue="gpu", confidence=0.95, recommended_batch_size=4)

    # Heavy complexity (tables, high image ratio) needs GPU
    if analysis.complexity == DocumentComplexity.HEAVY:
        return RouteDecision(queue="gpu", confidence=0.85, recommended_batch_size=4)

    # Mixed complexity - GPU for better quality
    if analysis.complexity == DocumentComplexity.MIXED:
        return RouteDecision(queue="gpu", confidence=0.80, recommended_batch_size=8)

    # Light complexity - check if CPU is safe
    cpu_confidence = _confidence_for_cpu(analysis)
    if cpu_confidence >= 0.85:
        return RouteDecision(queue="cpu", confidence=cpu_confidence, recommended_batch_size=12)

    # Light but low confidence - fall back to GPU
    return RouteDecision(queue="gpu", confidence=0.90, recommended_batch_size=12)


def _confidence_for_cpu(analysis: DocumentAnalysis) -> float:
    """Calculate confidence that CPU extraction will be accurate."""
    if not analysis.has_extractable_text:
        return 0.0  # Scanned, needs OCR

    issues = analysis.multi_column_pages
    confidence = 1.0 - (issues / max(analysis.page_count, 1))
    return max(0.0, min(1.0, confidence))


def get_dynamic_batch_size(analysis: DocumentAnalysis, vram_budget_gb: float = 14.0) -> int:
    """Calculate optimal batch size based on complexity and VRAM budget.

    VRAM estimates per page (empirical):
    - Heavy (image/table): ~0.3 GB per page in batch
    - Mixed: ~0.15 GB per page in batch
    - Light: ~0.08 GB per page in batch

    Args:
        analysis: DocumentAnalysis from analyze_document()
        vram_budget_gb: Available VRAM in GB (default 14GB for RTX 4070 Ti)

    Returns:
        Recommended batch size (capped at 16)
    """
    vram_per_page = {
        DocumentComplexity.HEAVY: 0.3,
        DocumentComplexity.MIXED: 0.15,
        DocumentComplexity.LIGHT: 0.08,
    }

    estimate = vram_per_page.get(analysis.complexity, 0.15)
    max_batch = int(vram_budget_gb / estimate)

    return min(max_batch, analysis.page_count, 16)  # Cap at 16
