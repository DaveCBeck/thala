"""Prompt loading and translation utilities."""

from typing import Optional, Tuple

from workflows.web_research.prompts.translator import get_translated_prompt


async def load_prompts_with_translation(
    system_template: str,
    user_template: str,
    language_config: Optional[dict],
    system_prompt_name: str,
    user_prompt_name: str,
) -> Tuple[str, str]:
    """Load prompts with optional translation for non-English languages."""
    if language_config and language_config.get("code", "en") != "en":
        system_prompt = await get_translated_prompt(
            system_template,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name=system_prompt_name,
        )
        user_prompt = await get_translated_prompt(
            user_template,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name=user_prompt_name,
        )
        return system_prompt, user_prompt
    return system_template, user_template


def get_language_config(state: dict) -> Optional[dict]:
    """Extract language config from state."""
    return state.get("language_config")
