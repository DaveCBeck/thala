"""Tests for IllustrateConfig fields and presets."""

import pytest
from pydantic import ValidationError

from workflows.output.illustrate.config import IllustrateConfig


class TestImagenSampleCount:
    def test_default_is_2(self):
        c = IllustrateConfig()
        assert c.imagen_sample_count == 2

    def test_min_1(self):
        c = IllustrateConfig(imagen_sample_count=1)
        assert c.imagen_sample_count == 1

    def test_max_4(self):
        c = IllustrateConfig(imagen_sample_count=4)
        assert c.imagen_sample_count == 4

    def test_rejects_0(self):
        with pytest.raises(ValidationError):
            IllustrateConfig(imagen_sample_count=0)

    def test_rejects_5(self):
        with pytest.raises(ValidationError):
            IllustrateConfig(imagen_sample_count=5)


class TestOvergenerationSurplus:
    def test_default_is_2(self):
        c = IllustrateConfig()
        assert c.overgeneration_surplus == 2

    def test_min_0(self):
        c = IllustrateConfig(overgeneration_surplus=0)
        assert c.overgeneration_surplus == 0

    def test_max_2(self):
        c = IllustrateConfig(overgeneration_surplus=2)
        assert c.overgeneration_surplus == 2

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            IllustrateConfig(overgeneration_surplus=-1)

    def test_rejects_3(self):
        with pytest.raises(ValidationError):
            IllustrateConfig(overgeneration_surplus=3)


class TestPresets:
    def test_quick(self):
        c = IllustrateConfig.quick()
        assert c.overgeneration_surplus == 0
        assert c.enable_editorial_review is False
        assert c.max_retries == 0
        assert c.imagen_sample_count == 1

    def test_balanced(self):
        c = IllustrateConfig.balanced()
        assert c.overgeneration_surplus == 1
        assert c.enable_editorial_review is True
        assert c.max_retries == 1
        assert c.imagen_sample_count == 2

    def test_quality(self):
        c = IllustrateConfig.quality()
        assert c.overgeneration_surplus == 2
        assert c.imagen_sample_count == 2

    def test_quick_allows_overrides(self):
        c = IllustrateConfig.quick(additional_image_count=1)
        assert c.additional_image_count == 1
        assert c.imagen_sample_count == 1  # preset default preserved

    def test_balanced_allows_overrides(self):
        c = IllustrateConfig.balanced(imagen_sample_count=1)
        assert c.imagen_sample_count == 1
        assert c.overgeneration_surplus == 1  # preset default preserved
