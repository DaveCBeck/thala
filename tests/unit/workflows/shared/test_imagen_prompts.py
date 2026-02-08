"""Tests for structured Imagen prompt builder (A4)."""

from workflows.shared.imagen_prompts import ImagenPromptStructure, build_imagen_prompt


class TestBuildImagenPrompt:
    """Test prompt building from structured input."""

    def test_all_parts_present(self):
        structure = ImagenPromptStructure(
            primary_subject="A golden retriever puppy",
            composition="close-up portrait",
            key_elements=["soft bokeh background", "warm sunlight"],
            style_and_mood="warm, nostalgic, soft golden hour lighting",
            context_setting="a meadow of wildflowers",
        )
        prompt = build_imagen_prompt(structure)
        assert prompt.startswith("A golden retriever puppy")
        assert "close-up portrait" in prompt
        assert "soft bokeh background" in prompt
        assert "warm sunlight" in prompt
        assert "meadow" in prompt
        assert "golden hour" in prompt

    def test_primary_subject_comes_first(self):
        structure = ImagenPromptStructure(
            primary_subject="A microscope on a lab bench",
            composition="overhead view",
            key_elements=["sample slides"],
            style_and_mood="clinical, cool blue lighting",
            context_setting="sterile laboratory",
        )
        prompt = build_imagen_prompt(structure)
        assert prompt.index("A microscope") < prompt.index("overhead view")
        assert prompt.index("A microscope") < prompt.index("laboratory")

    def test_empty_key_elements(self):
        structure = ImagenPromptStructure(
            primary_subject="Sunset over ocean",
            composition="wide panoramic shot",
            key_elements=[],
            style_and_mood="warm, dramatic",
            context_setting="rocky coastline",
        )
        prompt = build_imagen_prompt(structure)
        assert "Sunset over ocean" in prompt
        assert ",," not in prompt  # No empty part artifacts
