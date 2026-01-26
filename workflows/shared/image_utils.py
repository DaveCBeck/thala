"""Image generation utilities using Google Imagen with LLM-generated prompts."""

import logging
import os

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

# Image generation model
IMAGEN_MODEL = "imagen-4.0-ultra-generate-001"

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
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
        )

        # Truncate content if too long (keep first ~8000 chars)
        truncated_content = content[:8000] if len(content) > 8000 else content

        response = await llm.ainvoke(
            [
                {"role": "system", "content": IMAGE_PROMPT_SYSTEM},
                {
                    "role": "user",
                    "content": IMAGE_PROMPT_USER.format(
                        title=title, content=truncated_content
                    ),
                },
            ]
        )

        prompt = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()

        logger.info(f"Generated image prompt for '{title}': {prompt[:100]}...")
        return prompt

    except Exception as e:
        logger.error(f"Failed to generate image prompt for '{title}': {e}")
        return None


async def generate_article_header(
    title: str,
    content: str,
    theme: str | None = None,
    custom_prompt: str | None = None,
    aspect_ratio: str = "16:9",
) -> tuple[bytes | None, str | None]:
    """Generate an article header image using LLM-generated prompt + Imagen.

    Args:
        title: Article title
        content: Full article content (used by Sonnet to generate image prompt)
        theme: Optional theme description (unused in new flow, kept for compatibility)
        custom_prompt: If provided, skip LLM prompt generation and use this directly
        aspect_ratio: Imagen aspect ratio (default "16:9", also supports "1:1", "9:16", etc.)

    Returns:
        Tuple of (PNG image bytes, prompt used) or (None, None) if generation fails
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return None, None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None, None

    # Step 1: Use custom prompt or generate one using Sonnet
    if custom_prompt:
        prompt = custom_prompt
        logger.info(f"Using custom prompt for image generation: {prompt[:100]}...")
    else:
        prompt = await generate_image_prompt(title, content)
        if not prompt:
            logger.error(f"Failed to generate image prompt for '{title}'")
            return None, None

    # Step 2: Generate the image using Imagen
    try:
        client = genai.Client(api_key=api_key)

        response = await client.aio.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )

        if response.generated_images:
            image_bytes = response.generated_images[0].image.image_bytes
            logger.info(
                f"Generated header image for '{title}' "
                f"({len(image_bytes)} bytes)"
            )
            return image_bytes, prompt

        logger.warning(f"No image generated for '{title}' - response was empty")
        return None, prompt

    except Exception as e:
        logger.error(f"Failed to generate image for '{title}': {e}")
        return None, prompt
