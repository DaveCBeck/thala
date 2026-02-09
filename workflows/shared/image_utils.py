"""Image generation utilities using Google Imagen and Gemini with LLM-generated prompts."""

import asyncio
import logging
import os

from core.llm_broker import BatchPolicy
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

logger = logging.getLogger(__name__)

# Image generation models
IMAGEN_MODEL = "imagen-4.0-ultra-generate-001"
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"

# Lazy-initialized genai client (reused across calls)
_genai_client = None


def _get_genai_client():
    """Get or create the global genai client (lazy init)."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        from core.utils.async_http_client import register_cleanup

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        _genai_client = genai.Client(api_key=api_key)

        async def _close_genai():
            global _genai_client
            _genai_client = None

        register_cleanup("genai", _close_genai)
    return _genai_client


# System prompt for generating image prompts
IMAGE_PROMPT_SYSTEM = """You are an expert at writing prompts for AI image generation, specifically for Google's Imagen model.

## Your Task
Write a single image generation prompt for an article header image. The image should visually complement the article without literally illustrating it.

## Image Generation Best Practices

**Structure your prompt as a flowing narrative description with these elements:**
1. Start with clear intent: "A professional photograph of..." or "An editorial image of..."
2. Describe the visual concept and composition
3. Specify lighting precisely (direction, type, quality)
4. Include technical specs for photorealism: lens type, aperture
5. Add quality indicators: "4K," "professional," "editorial quality"
6. State aspect ratio: "16:9 aspect ratio"

**What works well:**
- Specific, concrete visual details over abstract concepts
- Photography terminology: camera angles, lens types (35mm, 85mm), lighting setups
- Directional lighting: "soft light from upper left" prevents flat, fake-looking images
- Quality modifiers: "4K," "HDR," "professional photography"
- Positive framing: describe what you WANT, not what to avoid

**What to avoid:**
- Keyword lists or bullet points - use flowing prose
- Vague descriptions that lead to generic results
- Conflicting styles or too many competing instructions
- Negative instructions ("no text") - these often backfire
- Overly literal illustrations of the article content

**For editorial/professional images:**
- Think like a professional photographer, not an artist
- Evocative and atmospheric, not literal
- Clean compositions with space for potential text overlay
- Muted, sophisticated color palettes
- Subtle imagery that suggests the theme without being on-the-nose

## Output Format
Return ONLY the image prompt, nothing else. No preamble, no explanation, no quotes around it.
The prompt should be 50-150 words of flowing narrative description."""

IMAGE_PROMPT_USER = """Write an image generation prompt for a header image for this article.

**Article Title:** {title}

**Article Content:**
{content}

Remember: Create an evocative, professional editorial image that complements the article's themes without literally illustrating them. The image should be a square format, 1:1 aspect ratio."""


async def generate_image_prompt(
    title: str,
    content: str,
) -> str | None:
    """Use Sonnet to generate an optimized image prompt based on the article.

    Args:
        title: Article title
        content: Full article content

    Returns:
        Generated image prompt string, or None if generation fails
    """
    try:
        # Truncate content if too long (keep first ~8000 chars)
        truncated_content = content[:8000] if len(content) > 8000 else content

        response = await invoke(
            tier=ModelTier.SONNET,
            system=IMAGE_PROMPT_SYSTEM,
            user=IMAGE_PROMPT_USER.format(title=title, content=truncated_content),
            config=InvokeConfig(
                max_tokens=500,
                batch_policy=BatchPolicy.PREFER_BALANCE,
            ),
        )

        prompt = (response.content if isinstance(response.content, str) else str(response.content)).strip()

        logger.info(f"Generated image prompt for '{title}': {prompt[:100]}...")
        return prompt

    except Exception as e:
        logger.error(f"Failed to generate image prompt for '{title}': {e}")
        return None


async def generate_article_header(
    title: str,
    content: str,
    custom_prompt: str | None = None,
    aspect_ratio: str = "16:9",
    sample_count: int = 4,
) -> tuple[bytes | None, str | None]:
    """Generate an article header image using LLM-generated prompt + Imagen.

    Generates multiple candidates and uses vision pair comparison to select
    the best one.

    Args:
        title: Article title
        content: Full article content (used by Sonnet to generate image prompt)
        custom_prompt: If provided, skip LLM prompt generation and use this directly
        aspect_ratio: Imagen aspect ratio (default "16:9", also supports "1:1", "9:16", etc.)
        sample_count: Number of image candidates to generate (default 4)

    Returns:
        Tuple of (PNG image bytes, prompt used) or (None, None) if generation fails
    """
    try:
        from google.genai import types
    except ImportError:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return None, None

    try:
        client = _get_genai_client()
    except ValueError:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None, None

    # Step 1: Use custom prompt (structured via Imagen optimizer) or generate one
    if custom_prompt:
        from workflows.shared.imagen_prompts import structure_brief_for_imagen

        prompt = await structure_brief_for_imagen(custom_prompt)
        logger.info(f"Structured prompt for image generation: {prompt[:100]}...")
    else:
        prompt = await generate_image_prompt(title, content)
        if not prompt:
            logger.error(f"Failed to generate image prompt for '{title}'")
            return None, None

    # Step 2: Generate images using Imagen (semaphore limits concurrent API calls)
    try:
        from core.task_queue.rate_limits import get_imagen_semaphore

        async with get_imagen_semaphore():
            response = await client.aio.models.generate_images(
                model=IMAGEN_MODEL,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=sample_count,
                    aspect_ratio=aspect_ratio,
                ),
            )

        candidates = [
            img.image.image_bytes for img in (response.generated_images or []) if img.image and img.image.image_bytes
        ]

        if not candidates:
            logger.warning(f"No images generated for '{title}' - response was empty")
            return None, prompt

        if len(candidates) == 1:
            logger.info(f"Generated 1 header image for '{title}' ({len(candidates[0])} bytes)")
            return candidates[0], prompt

        # Step 3: Vision pair comparison to select best candidate
        logger.info(f"Generated {len(candidates)} candidates for '{title}', selecting best...")
        best = await _select_best_imagen_candidate(candidates, prompt, custom_prompt or "")
        return best, prompt

    except Exception as e:
        logger.error(f"Failed to generate image for '{title}': {e}")
        return None, prompt


async def _select_best_imagen_candidate(
    candidates: list[bytes],
    prompt: str,
    brief: str,
) -> bytes:
    """Select the best Imagen candidate via vision pair comparison.

    Falls back to first candidate if vision comparison fails.
    """
    try:
        from workflows.shared.vision_comparison import vision_pair_select

        criteria = brief if brief else prompt
        best_idx = await vision_pair_select(candidates, criteria)
        logger.info(f"Vision selected Imagen candidate {best_idx + 1} of {len(candidates)}")
        return candidates[best_idx]
    except Exception as e:
        logger.warning(f"Imagen vision selection failed, using first candidate: {e}")
        return candidates[0]


DIAGRAM_PROMPT_PREFIX = (
    "Generate a clear, professional diagram image with sharp, legible text. "
    "Use clean lines, consistent spacing, and high contrast. "
    "The diagram should be informative and visually organized.\n\n"
)


async def generate_diagram_image(
    brief: str,
    aspect_ratio: str = "3:2",
    num_candidates: int = 2,
) -> tuple[bytes | None, str | None]:
    """Generate a diagram image using Gemini 3 Pro image generation.

    Gemini 3 Pro is optimized for sharp, legible text and diagrams at up to
    2K/4K resolution — replacing the previous Mermaid/Graphviz/SVG pipeline.

    Generates multiple candidates in parallel and uses vision pair comparison
    to select the best one.

    Args:
        brief: Natural-language diagram brief from plan_briefs.
        aspect_ratio: Image aspect ratio (default "3:2" for 900x600 diagram shape).
        num_candidates: Number of parallel generation calls (default 2).

    Returns:
        Tuple of (PNG image bytes, prompt used) or (None, None) if generation fails.
    """
    try:
        from google.genai import types
    except ImportError:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return None, None

    try:
        client = _get_genai_client()
    except ValueError:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None, None

    prompt = DIAGRAM_PROMPT_PREFIX + brief

    async def _generate_one() -> bytes | None:
        """Single Gemini image generation call."""
        from core.task_queue.rate_limits import get_imagen_semaphore

        async with get_imagen_semaphore():
            response = await client.aio.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            )

        # Extract image bytes from response parts
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        return part.inline_data.data
            # Log what we got instead of an image
            text_parts = [
                p.text[:100] for p in (candidate.content.parts if candidate.content and candidate.content.parts else []) if p.text
            ]
            logger.warning(f"Gemini returned no image. finish_reason={candidate.finish_reason}, text={text_parts}")
        else:
            logger.warning(f"Gemini returned no candidates. prompt_feedback={response.prompt_feedback}")
        return None

    try:
        results = await asyncio.gather(
            *[_generate_one() for _ in range(num_candidates)],
            return_exceptions=True,
        )

        candidates = [r for r in results if isinstance(r, bytes)]

        if not candidates:
            errors = [r for r in results if isinstance(r, Exception)]
            logger.warning(f"No diagram images generated — {len(errors)} errors: {errors[:2]}")
            return None, prompt

        if len(candidates) == 1:
            logger.info(f"Generated 1 diagram candidate ({len(candidates[0])} bytes)")
            return candidates[0], prompt

        logger.info(f"Generated {len(candidates)} diagram candidates, selecting best...")
        best = await _select_best_diagram_candidate(candidates, prompt, brief)
        return best, prompt

    except Exception as e:
        logger.error(f"Diagram image generation failed: {e}")
        return None, prompt


async def _select_best_diagram_candidate(
    candidates: list[bytes],
    prompt: str,
    brief: str,
) -> bytes:
    """Select the best diagram candidate via vision pair comparison.

    Falls back to first candidate if vision comparison fails.
    """
    try:
        from workflows.shared.vision_comparison import vision_pair_select

        criteria = brief if brief else prompt
        best_idx = await vision_pair_select(candidates, criteria)
        logger.info(f"Vision selected diagram candidate {best_idx + 1} of {len(candidates)}")
        return candidates[best_idx]
    except Exception as e:
        logger.warning(f"Diagram vision selection failed, using first candidate: {e}")
        return candidates[0]
