"""Mermaid diagram generation with validate-repair-render loop.

Generates Mermaid diagram code via LLM, validates syntax, repairs
errors (up to 2 attempts), and renders to PNG using the mmdc package.
"""

import asyncio
import logging

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from .schemas import DiagramConfig, DiagramResult

logger = logging.getLogger(__name__)

MERMAID_GENERATION_SYSTEM = """You are an expert at creating Mermaid diagrams. Generate clean, readable Mermaid code.

RULES:
- Use quotes around node labels that contain special characters or spaces
- Always close subgraphs with 'end'
- Use valid edge syntax: --> for directed, --- for undirected
- Keep node labels concise (under 40 characters)
- Use subgraphs to group related concepts
- Maximum 15-20 nodes for readability
- Use meaningful node IDs (not just A, B, C)
- For flowcharts, prefer 'graph TD' (top-down) or 'graph LR' (left-right)

Output ONLY the Mermaid code, no markdown fences, no explanation."""

MERMAID_GENERATION_USER = """Generate a Mermaid diagram for the following concept.

{instructions}

Output ONLY the Mermaid code."""

MERMAID_REPAIR_SYSTEM = """You are fixing a Mermaid diagram that has syntax errors.
Fix the errors while preserving the diagram's intent and structure.

Common fixes:
- Add quotes around labels with special characters (parentheses, colons, etc.)
- Close unclosed subgraphs with 'end'
- Fix edge syntax (use --> not ->)
- Remove invalid characters from node IDs
- Fix indentation issues

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
        return _failure("LLM failed to generate Mermaid code")

    # Step 2: Validate + repair loop (initial + 2 repair attempts)
    is_valid = False
    errors = ""
    for attempt in range(3):
        is_valid, errors = _validate_mermaid(mermaid_code)
        if is_valid:
            break
        if attempt < 2:
            logger.info(f"Mermaid validation failed (attempt {attempt + 1}), repairing: {errors}")
            repaired = await _llm_repair_mermaid(mermaid_code, errors)
            if repaired:
                mermaid_code = repaired

    if not is_valid:
        return _failure(f"Mermaid validation failed after repairs: {errors}")

    # Step 3: Render to PNG
    png_bytes = await _render_mermaid_to_png(mermaid_code, width=config.width, background=config.background_color)
    if not png_bytes:
        return _failure("Mermaid rendering failed")

    logger.info(f"Mermaid diagram generated ({len(png_bytes)} bytes PNG)")
    return DiagramResult(
        svg_bytes=mermaid_code.encode("utf-8"),  # Store source code as "svg_bytes"
        png_bytes=png_bytes,
        analysis=None,
        overlap_check=None,
        generation_attempts=1,
        success=True,
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

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:])

        return content.strip()
    except Exception as e:
        logger.error(f"Mermaid generation failed: {e}")
        return None


def _validate_mermaid(code: str) -> tuple[bool, str]:
    """Validate Mermaid syntax.

    Uses mmdc to attempt a parse — if it fails, the error message
    is returned for the repair loop.
    """
    try:
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

        # Strip markdown fences
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:])

        return content.strip()
    except Exception as e:
        logger.error(f"Mermaid repair failed: {e}")
        return None


async def _render_mermaid_to_png(code: str, width: int = 800, background: str = "#ffffff") -> bytes | None:
    """Render Mermaid code to PNG using mmdc. Runs in thread to avoid blocking."""

    def _render() -> bytes | None:
        try:
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
    from workflows.shared.vision_comparison import vision_pair_select

    instructions = custom_instructions or analysis

    tasks = [generate_mermaid_diagram(instructions, config, custom_instructions) for _ in range(num_candidates)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if isinstance(r, DiagramResult) and r.success and r.png_bytes]

    if not successful:
        return _failure("All Mermaid candidates failed")
    if len(successful) == 1:
        return successful[0]

    # Vision pair comparison to select best
    png_list = [r.png_bytes for r in successful]
    try:
        best_idx = await vision_pair_select(png_list, instructions)
    except Exception as e:
        logger.warning(f"Vision selection failed, using first candidate: {e}")
        best_idx = 0

    selected = successful[best_idx]
    selected.selected_candidate = best_idx + 1
    selected.generation_attempts = len(successful)
    return selected


def _failure(error: str) -> DiagramResult:
    """Create a failed DiagramResult."""
    return DiagramResult(
        svg_bytes=None,
        png_bytes=None,
        analysis=None,
        overlap_check=None,
        generation_attempts=1,
        success=False,
        error=error,
    )


__all__ = [
    "generate_mermaid_diagram",
    "generate_mermaid_with_selection",
]
