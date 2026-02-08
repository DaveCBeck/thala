"""Graphviz DOT diagram generation with validate-repair-render loop.

Generates DOT code via LLM, validates by attempting to render with
the graphviz Python package (requires system `dot` binary), and
repairs errors (up to 2 attempts).
"""

import asyncio
import logging
import re

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import InvokeConfig, ModelTier, invoke

from .schemas import DiagramConfig, DiagramResult
from .validation import strip_code_fences

logger = logging.getLogger(__name__)

_FORBIDDEN_DOT_ATTRS = re.compile(
    r'\b(image|shapefile|fontpath|imagepath)\s*=', re.IGNORECASE
)


def _sanitize_dot_code(code: str) -> str:
    """Reject DOT code containing file-access attributes."""
    if _FORBIDDEN_DOT_ATTRS.search(code):
        raise ValueError("DOT code contains forbidden file-access attributes")
    return code

GRAPHVIZ_GENERATION_SYSTEM = """You are an expert at creating Graphviz DOT diagrams. Generate clean, readable DOT code.

RULES:
- Use valid DOT syntax (digraph for directed, graph for undirected)
- Quote node labels that contain special characters
- Use readable node IDs
- Keep labels concise (under 40 characters)
- Use subgraphs (cluster_) to group related concepts
- Set rankdir=TB (top-bottom) or rankdir=LR (left-right) based on content
- Use shape=box for processes, shape=diamond for decisions, shape=ellipse for start/end
- Maximum 15-20 nodes for readability
- Include graph attributes: fontname, fontsize, bgcolor
- Include node defaults: style=filled, fillcolor, fontname

Output ONLY the DOT code, no markdown fences, no explanation."""

GRAPHVIZ_GENERATION_USER = """Generate a Graphviz DOT diagram for the following concept.

<instructions>
{instructions}
</instructions>

Output ONLY the DOT code."""

GRAPHVIZ_REPAIR_SYSTEM = """You are fixing a Graphviz DOT diagram that has syntax errors.
Fix the errors while preserving the diagram's intent and structure.

Common fixes:
- Add missing semicolons
- Fix unclosed braces or quotes
- Correct attribute syntax (use = not :)
- Quote labels with special characters
- Fix edge syntax (-> for digraph, -- for graph)

Output ONLY the corrected DOT code, no explanation."""


async def generate_graphviz_diagram(
    analysis: str,
    config: DiagramConfig,
    custom_instructions: str = "",
) -> DiagramResult:
    """Generate a Graphviz diagram with validate-repair loop.

    Args:
        analysis: Content analysis or description to diagram
        config: Diagram configuration
        custom_instructions: Additional instructions for the LLM

    Returns:
        DiagramResult with png_bytes on success
    """
    instructions = custom_instructions or analysis

    dot_code = await _llm_generate_dot(instructions)
    if not dot_code:
        return DiagramResult.failure("LLM failed to generate DOT code")

    # Validate-repair loop (initial + 2 repair attempts)
    for attempt in range(3):
        png_bytes, error = await _render_dot_to_png(dot_code, config)
        if png_bytes:
            logger.info(f"Graphviz diagram generated ({len(png_bytes)} bytes PNG)")
            return DiagramResult(
                svg_bytes=None,
                png_bytes=png_bytes,
                analysis=None,
                overlap_check=None,
                generation_attempts=1,
                success=True,
                source_code=dot_code,
            )
        if attempt < 2:
            logger.info(f"Graphviz rendering failed (attempt {attempt + 1}), repairing: {error}")
            repaired = await _llm_repair_dot(dot_code, error or "Unknown error")
            if repaired:
                dot_code = repaired

    return DiagramResult.failure(f"Graphviz rendering failed after repairs: {error}")


async def _llm_generate_dot(instructions: str) -> str | None:
    """Use LLM to generate DOT code from instructions."""
    try:
        response = await invoke(
            tier=ModelTier.SONNET,
            system=GRAPHVIZ_GENERATION_SYSTEM,
            user=GRAPHVIZ_GENERATION_USER.format(instructions=instructions),
            config=InvokeConfig(
                max_tokens=2000,
                batch_policy=BatchPolicy.PREFER_SPEED,
            ),
        )
        content = (response.content if isinstance(response.content, str) else str(response.content)).strip()
        return strip_code_fences(content)
    except Exception as e:
        logger.error(f"DOT generation failed: {e}")
        return None


async def _llm_repair_dot(code: str, errors: str) -> str | None:
    """Use LLM to repair DOT syntax errors."""
    try:
        response = await invoke(
            tier=ModelTier.SONNET,
            system=GRAPHVIZ_REPAIR_SYSTEM,
            user=f"Fix this DOT diagram:\n\nERRORS:\n{errors}\n\nCODE:\n{code}",
            config=InvokeConfig(max_tokens=2000),
        )
        content = (response.content if isinstance(response.content, str) else str(response.content)).strip()
        return strip_code_fences(content)
    except Exception as e:
        logger.error(f"DOT repair failed: {e}")
        return None


async def _render_dot_to_png(dot_code: str, config: DiagramConfig) -> tuple[bytes | None, str | None]:
    """Render DOT code to PNG. Runs in thread to avoid blocking."""
    # Sanitize before DOT code reaches graphviz.Source()
    try:
        _sanitize_dot_code(dot_code)
    except ValueError as e:
        return None, str(e)

    def _render() -> tuple[bytes | None, str | None]:
        import graphviz  # lazy: only needed when actually rendering

        try:
            source = graphviz.Source(dot_code)
            png_bytes = source.pipe(format="png")
            return png_bytes, None
        except graphviz.CalledProcessError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)

    return await asyncio.to_thread(_render)


async def generate_graphviz_with_selection(
    analysis: str,
    config: DiagramConfig,
    custom_instructions: str = "",
    num_candidates: int = 3,
) -> DiagramResult:
    """Generate multiple Graphviz candidates, select best via vision.

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
        generator_fn=generate_graphviz_diagram,
        analysis=analysis,
        config=config,
        custom_instructions=custom_instructions,
        num_candidates=num_candidates,
        engine_name="Graphviz",
    )


__all__ = [
    "generate_graphviz_diagram",
    "generate_graphviz_with_selection",
]
