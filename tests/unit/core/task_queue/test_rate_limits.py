"""Tests for rate_limits.py semaphore lazy initialization."""

import asyncio
from unittest.mock import patch

import pytest

import core.task_queue.rate_limits as rate_limits_mod
from core.task_queue.rate_limits import get_imagen_semaphore, get_openalex_semaphore


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-level semaphore singletons between tests."""
    rate_limits_mod._imagen_semaphore = None
    rate_limits_mod._openalex_semaphore = None
    yield
    rate_limits_mod._imagen_semaphore = None
    rate_limits_mod._openalex_semaphore = None


# ---------------------------------------------------------------------------
# get_imagen_semaphore
# ---------------------------------------------------------------------------

class TestGetImagenSemaphore:
    """Tests for the Imagen semaphore factory."""

    def test_lazy_init_creates_semaphore(self):
        """First call creates a Semaphore; second call returns the same instance."""
        first = get_imagen_semaphore()
        second = get_imagen_semaphore()

        assert isinstance(first, asyncio.Semaphore)
        assert first is second

    def test_default_limit_is_10(self):
        """Default concurrency limit for Imagen is 10."""
        sem = get_imagen_semaphore()

        # asyncio.Semaphore stores the initial value in _value
        assert sem._value == 10

    def test_env_var_override(self):
        """THALA_IMAGEN_CONCURRENCY env var changes the limit."""
        with patch.dict("os.environ", {"THALA_IMAGEN_CONCURRENCY": "3"}):
            sem = get_imagen_semaphore()

        assert sem._value == 3


# ---------------------------------------------------------------------------
# get_openalex_semaphore
# ---------------------------------------------------------------------------

class TestGetOpenAlexSemaphore:
    """Tests for the OpenAlex semaphore factory."""

    def test_lazy_init_creates_semaphore(self):
        """First call creates a Semaphore; second call returns the same instance."""
        first = get_openalex_semaphore()
        second = get_openalex_semaphore()

        assert isinstance(first, asyncio.Semaphore)
        assert first is second

    def test_default_limit_is_20(self):
        """Default concurrency limit for OpenAlex is 20."""
        sem = get_openalex_semaphore()

        assert sem._value == 20

    def test_env_var_override(self):
        """THALA_OPENALEX_CONCURRENCY env var changes the limit."""
        with patch.dict("os.environ", {"THALA_OPENALEX_CONCURRENCY": "7"}):
            sem = get_openalex_semaphore()

        assert sem._value == 7


# ---------------------------------------------------------------------------
# Cross-semaphore independence
# ---------------------------------------------------------------------------

class TestSemaphoreIndependence:
    """Tests that the two semaphores are independent."""

    def test_imagen_and_openalex_are_different_instances(self):
        """Imagen and OpenAlex return distinct semaphore objects."""
        imagen = get_imagen_semaphore()
        openalex = get_openalex_semaphore()

        assert imagen is not openalex

    def test_resetting_one_does_not_affect_other(self):
        """Resetting the Imagen global does not clear the OpenAlex global."""
        imagen = get_imagen_semaphore()
        openalex = get_openalex_semaphore()

        # Reset only Imagen
        rate_limits_mod._imagen_semaphore = None

        new_imagen = get_imagen_semaphore()
        same_openalex = get_openalex_semaphore()

        assert new_imagen is not imagen
        assert same_openalex is openalex
