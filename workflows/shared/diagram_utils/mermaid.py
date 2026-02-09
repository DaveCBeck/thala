"""Mermaid diagram generation with validate-repair-render loop.

Generates Mermaid diagram code via LLM, validates syntax, repairs
errors (up to 2 attempts), and renders to PNG using the mmdc package.
"""

import asyncio
import logging
import re

from core.llm_broker import BatchPolicy
from core.task_queue.rate_limits import get_mmdc_semaphore
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from .schemas import DiagramConfig, DiagramResult
from .validation import strip_code_fences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mermaid code sanitization — PhantomJS XSS mitigation
# ---------------------------------------------------------------------------
# mmdc uses PhantomJS 2.1.1 (abandoned since 2016, known CVEs including
# CVE-2019-17221).  Mermaid supports HTML in node labels, so LLM-generated
# code could inject <script>, event handlers, etc. that execute inside the
# PhantomJS context.  We reject any code containing dangerous patterns
# *before* it reaches the renderer.

# Patterns that could trigger XSS in PhantomJS renderer
_DANGEROUS_HTML_PATTERN = re.compile(
    r'<\s*(?:script|iframe|object|embed|link|style|img\b[^>]*\bonerror)[^>]*>',
    re.IGNORECASE,
)
_DANGEROUS_ATTR_PATTERN = re.compile(
    r'\bon\w+\s*=',
    re.IGNORECASE,
)


def _sanitize_mermaid_code(code: str) -> str:
    """Strip dangerous HTML that could exploit PhantomJS renderer (CVE-2019-17221 et al.)."""
    if _DANGEROUS_HTML_PATTERN.search(code) or _DANGEROUS_ATTR_PATTERN.search(code):
        raise ValueError("Mermaid code contains potentially dangerous HTML content")
    return code


MERMAID_GENERATION_SYSTEM = """You are an expert at creating Mermaid diagrams. Generate clean, readable Mermaid code.

RULES:
- Use quotes around node labels that contain special characters or spaces
- Use valid edge syntax: --> for directed, --- for undirected
- Keep node labels concise (under 40 characters)
- Maximum 12-15 nodes for readability
- Use meaningful node IDs (not just A, B, C)
- For flowcharts, prefer 'graph TD' (top-down) or 'graph LR' (left-right)
- Use edge labels (e.g. -->|"label"|) to annotate relationships
- DO NOT use 'subgraph' blocks — they are not supported by the renderer
- Keep the diagram FLAT: all nodes at the same level, connected by edges

Output ONLY the Mermaid code, no markdown fences, no explanation."""

MERMAID_GENERATION_USER = """Generate a Mermaid diagram for the following concept.

<instructions>
{instructions}
</instructions>

Output ONLY the Mermaid code."""

MERMAID_REPAIR_SYSTEM = """You are fixing a Mermaid diagram that has syntax errors.
Fix the errors while preserving the diagram's intent and structure.

Common fixes:
- Add quotes around labels with special characters (parentheses, colons, etc.)
- Fix edge syntax (use --> not ->)
- Remove invalid characters from node IDs
- Fix indentation issues
- Remove any 'subgraph' blocks (not supported) — flatten into regular nodes and edges

Output ONLY the corrected Mermaid code, no explanation."""


async def generate_mermaid_diagram(
    analysis: str,
    config: DiagramConfig,
    custom_instructions: str = "",
) -> DiagramResult:
    """Generate a Mermaid diagram with validate-repair loop.

    Args:
        analysis: Content analysis or description to diagram
        config: Diagram configuration (width used for rendering)
        custom_instructions: Additional instructions for the LLM

    Returns:
        DiagramResult with png_bytes on success
    """
    instructions = custom_instructions or analysis

    # Step 1: LLM generates Mermaid code
    mermaid_code = await _llm_generate_mermaid(instructions)
    if not mermaid_code:
        return DiagramResult.failure("LLM failed to generate Mermaid code")

    # Step 1.5: Sanitize LLM output before it reaches PhantomJS
    try:
        mermaid_code = _sanitize_mermaid_code(mermaid_code)
    except ValueError as exc:
        return DiagramResult.failure(str(exc))

    # Step 2: Validate + repair loop (initial + 2 repair attempts)
    is_valid = False
    errors = ""
    for attempt in range(3):
        async with get_mmdc_semaphore():
            is_valid, errors = await asyncio.to_thread(_validate_mermaid, mermaid_code)
        if is_valid:
            break
        if attempt < 2:
            logger.info(f"Mermaid validation failed (attempt {attempt + 1}), repairing: {errors}")
            repaired = await _llm_repair_mermaid(mermaid_code, errors)
            if repaired:
                try:
                    mermaid_code = _sanitize_mermaid_code(repaired)
                except ValueError as exc:
                    return DiagramResult.failure(str(exc))

    if not is_valid:
        return DiagramResult.failure(f"Mermaid validation failed after repairs: {errors}")

    # Step 3: Render to PNG
    png_bytes = await _render_mermaid_to_png(mermaid_code, width=config.width, background=config.background_color)
    if not png_bytes:
        return DiagramResult.failure("Mermaid rendering failed")

    logger.info(f"Mermaid diagram generated ({len(png_bytes)} bytes PNG)")
    return DiagramResult(
        svg_bytes=None,
        png_bytes=png_bytes,
        analysis=None,
        overlap_check=None,
        generation_attempts=1,
        success=True,
        source_code=mermaid_code,
    )


async def _llm_generate_mermaid(instructions: str) -> str | None:
    """Use LLM to generate Mermaid code from instructions."""
    try:
        response = await invoke(
            tier=ModelTier.SONNET,
            system=MERMAID_GENERATION_SYSTEM,
            user=MERMAID_GENERATION_USER.format(instructions=instructions),
            config=InvokeConfig(
                max_tokens=2000,
                batch_policy=BatchPolicy.PREFER_SPEED,
            ),
        )
        content = (response.content if isinstance(response.content, str) else str(response.content)).strip()
        return strip_code_fences(content)
    except Exception as e:
        logger.error(f"Mermaid generation failed: {e}")
        return None


def _validate_mermaid(code: str) -> tuple[bool, str]:
    """Validate Mermaid syntax.

    Uses mmdc to attempt a parse — if it fails, the error message
    is returned for the repair loop.
    """
    try:
        # NOTE: mmdc uses PhantomJS (abandoned, known CVEs). Input is sanitized above.
        # Consider migrating to @mermaid-js/mermaid-cli (Playwright-based) in future.
        from mmdc import MermaidConverter

        converter = MermaidConverter(timeout=15)
        # Try to convert to SVG as a validation step (cheaper than PNG)
        result = converter.convert(code)
        if result is not None:
            return True, ""
        return False, "mmdc returned None (likely syntax error)"
    except Exception as e:
        return False, str(e)


async def _llm_repair_mermaid(code: str, errors: str) -> str | None:
    """Use LLM to repair Mermaid syntax errors."""
    try:
        response = await invoke(
            tier=ModelTier.SONNET,
            system=MERMAID_REPAIR_SYSTEM,
            user=f"Fix this Mermaid diagram:\n\nERRORS:\n{errors}\n\nCODE:\n{code}",
            config=InvokeConfig(max_tokens=2000),
        )
        content = (response.content if isinstance(response.content, str) else str(response.content)).strip()
        return strip_code_fences(content)
    except Exception as e:
        logger.error(f"Mermaid repair failed: {e}")
        return None


async def _render_mermaid_to_png(code: str, width: int = 800, background: str = "#ffffff") -> bytes | None:
    """Render Mermaid code to PNG using mmdc. Runs in thread to avoid blocking."""

    def _render() -> bytes | None:
        try:
            # NOTE: mmdc uses PhantomJS (abandoned, known CVEs). Input is sanitized above.
            # Consider migrating to @mermaid-js/mermaid-cli (Playwright-based) in future.
            from mmdc import MermaidConverter

            converter = MermaidConverter(timeout=30)
            png_bytes = converter.to_png(
                code,
                width=width,
                background=background,
            )
            return png_bytes
        except Exception as e:
            logger.error(f"Mermaid rendering failed: {e}")
            return None

    async with get_mmdc_semaphore():
        return await asyncio.to_thread(_render)


async def generate_mermaid_with_selection(
    analysis: str,
    config: DiagramConfig,
    custom_instructions: str = "",
    num_candidates: int = 3,
) -> DiagramResult:
    """Generate multiple Mermaid candidates, select best via vision.

    Args:
        analysis: Content analysis or description
        config: Diagram configuration
        custom_instructions: Additional instructions
        num_candidates: Number of parallel candidates to generate

    Returns:
        Best DiagramResult from candidates
    """
    from .schemas import generate_with_selection

    return await generate_with_selection(
        generator_fn=generate_mermaid_diagram,
        analysis=analysis,
        config=config,
        custom_instructions=custom_instructions,
        num_candidates=num_candidates,
        engine_name="Mermaid",
    )


__all__ = [
    "generate_mermaid_diagram",
    "generate_mermaid_with_selection",
]
