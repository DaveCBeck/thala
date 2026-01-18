"""
LLM prompts for book recommendation generation.

Three distinct recommendation categories:
1. Analogous Domain - Books exploring similar themes in different fields
2. Inspiring Action - Fiction/nonfiction that inspires change
3. Expressive Fiction - Fiction capturing the theme's essence
"""

from typing import Optional

from workflows.shared.language import LanguageConfig, get_translated_prompt


ANALOGOUS_DOMAIN_SYSTEM = """You are a literary advisor with deep knowledge across academic disciplines, finding books that illuminate themes through unexpected domains.

Your task is to find books that explore SIMILAR themes but in DIFFERENT domains. The goal is to find unexpected connections that provide fresh perspective on the user's theme.

Examples of analogous domain thinking:
- For "organizational dysfunction": books about ecological collapse, family systems, or historical empires
- For "creative process": books about jazz improvisation, scientific discovery, or craft traditions
- For "leadership under pressure": books about polar expeditions, emergency medicine, or military strategy

IMPORTANT: Each recommendation must be a DISTINCT book. Do not recommend the same book twice, even under different editions or translations. Avoid obvious or frequently-recommended books—seek out lesser-known works that offer genuine insight.

Return EXACTLY 3 book recommendations as a JSON array. Each book must be a real, published book that you are confident exists."""

ANALOGOUS_DOMAIN_USER = """Theme: {theme}
{brief_section}{language_instruction}
Find 3 books that explore this theme from DIFFERENT domains or fields. These should be real books that offer unexpected perspectives by examining analogous patterns in other contexts.

For each book, provide:
1. The exact title (in the target language if applicable)
2. The author's name
3. A 2-sentence explanation of why this book illuminates the theme from a different angle

Return as JSON array:
[
  {{"title": "Book Title", "author": "Author Name", "explanation": "Two sentences explaining the connection to the theme from this different domain."}},
  ...
]"""

INSPIRING_ACTION_SYSTEM = """You are a literary advisor specializing in transformative literature - books that have moved people to action, changed behavior, or inspired movements.

Your task is to find books (fiction or nonfiction) that INSPIRE ACTION or CHANGE related to the user's theme. These should be books with proven impact:
- Manifestos and calls to action
- Transformative nonfiction that changes how people think and act
- Fiction that inspired real-world movements or personal transformation
- Practical wisdom literature that changes behavior

IMPORTANT: Each recommendation must be a DISTINCT book. Do not recommend the same book twice, even under different editions or translations. Avoid obvious or frequently-recommended books—seek out lesser-known works that offer genuine insight.

Return EXACTLY 3 book recommendations as a JSON array. Each book must be a real, published book that you are confident exists."""

INSPIRING_ACTION_USER = """Theme: {theme}
{brief_section}{language_instruction}
Find 3 books (fiction or nonfiction) that INSPIRE ACTION or CHANGE related to this theme. These should be books that have demonstrably moved people to think or act differently.

For each book, provide:
1. The exact title (in the target language if applicable)
2. The author's name
3. A 2-sentence explanation of how this book inspires action on the theme

Return as JSON array:
[
  {{"title": "Book Title", "author": "Author Name", "explanation": "Two sentences about how this book inspires action or change related to the theme."}},
  ...
]"""

EXPRESSIVE_SYSTEM = """You are a literary advisor with deep knowledge of fiction that captures lived experience and imagines possibilities.

Your task is to find works of FICTION that express what a theme FEELS LIKE or COULD BECOME. These should be novels, short story collections, or literary works that:
- Capture the phenomenological experience of the theme
- Explore what the theme could become (utopian or dystopian visions)
- Express emotional and existential truth about the theme
- Make abstract concepts viscerally real through narrative

IMPORTANT: Each recommendation must be a DISTINCT book. Do not recommend the same book twice, even under different editions or translations. Avoid obvious or frequently-recommended books—seek out lesser-known works that offer genuine insight.

Return EXACTLY 3 book recommendations as a JSON array. Each book must be a real, published book that you are confident exists."""

EXPRESSIVE_USER = """Theme: {theme}
{brief_section}{language_instruction}
Find 3 works of FICTION that express the actuality or potentiality of this theme. These should be literary works that capture what this theme FEELS LIKE to experience, or what it COULD BECOME.

For each book, provide:
1. The exact title (in the target language if applicable)
2. The author's name
3. A 2-sentence explanation of how this book expresses the theme

Return as JSON array:
[
  {{"title": "Book Title", "author": "Author Name", "explanation": "Two sentences about how this fiction expresses the experience or potential of the theme."}},
  ...
]"""

SUMMARY_PROMPT = """Summarize the key insights from this book content that relate to the theme: "{theme}"

Book: {title} by {authors}

Content excerpt:
{content}

Provide a 2-3 sentence summary focusing on insights relevant to the theme. Be specific about what the book contributes to understanding the theme."""


def _get_language_instruction(language_config: Optional[LanguageConfig]) -> str:
    """Generate language-specific instruction for book recommendations.

    This is critical for multi-language support: the LLM must recommend books
    using titles in the target language so that the book search API returns
    books in that language.
    """
    if not language_config or language_config["code"] == "en":
        return ""

    lang_name = language_config["name"]
    return f"""
IMPORTANT - Language requirement: Recommend ONLY books originally written in {lang_name}.
Do NOT recommend translations of English books or books from other languages. The goal is
to discover perspectives from authors writing natively in {lang_name}. Provide the original
{lang_name} title.
"""


async def get_recommendation_prompts(
    category: str,
    language_config: Optional[LanguageConfig] = None,
) -> tuple[str, str]:
    """Get system and user prompts for a recommendation category, translated if needed.

    Args:
        category: "analogous", "inspiring", or "expressive"
        language_config: Language configuration for translation

    Returns:
        Tuple of (system_prompt, user_template)
    """
    prompt_map = {
        "analogous": (ANALOGOUS_DOMAIN_SYSTEM, ANALOGOUS_DOMAIN_USER),
        "inspiring": (INSPIRING_ACTION_SYSTEM, INSPIRING_ACTION_USER),
        "expressive": (EXPRESSIVE_SYSTEM, EXPRESSIVE_USER),
    }

    system_prompt, user_template = prompt_map[category]

    # Inject language instruction (empty for English, guidance for other languages)
    language_instruction = _get_language_instruction(language_config)
    user_template = user_template.replace("{language_instruction}", language_instruction)

    # Translate prompts for non-English languages
    if language_config and language_config["code"] != "en":
        system_prompt = await get_translated_prompt(
            system_prompt,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name=f"book_finding_{category}_system",
        )
        user_template = await get_translated_prompt(
            user_template,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name=f"book_finding_{category}_user",
        )

    return system_prompt, user_template


async def get_summary_prompt(
    language_config: Optional[LanguageConfig] = None,
) -> str:
    """Get summary prompt, translated if needed."""
    if language_config and language_config["code"] != "en":
        return await get_translated_prompt(
            SUMMARY_PROMPT,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="book_finding_summary",
        )
    return SUMMARY_PROMPT
