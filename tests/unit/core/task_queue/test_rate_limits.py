"""Tests for rate_limits.py semaphore lazy initialization, daily tracker, and RPM limiter."""

import asyncio
import json
from unittest.mock import patch

import pytest

import core.task_queue.rate_limits as rate_limits_mod
from core.task_queue.rate_limits import (
    ImagenDailyTracker,
    ImagenRPMLimiter,
    get_imagen_daily_tracker,
    get_imagen_rpm_limiter,
    get_imagen_semaphore,
    get_openalex_semaphore,
    reset_rate_limiters,
)


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-level singletons between tests."""
    rate_limits_mod._imagen_semaphore = None
    rate_limits_mod._openalex_semaphore = None
    rate_limits_mod._mmdc_semaphore = None
    rate_limits_mod._imagen_daily_tracker = None
    rate_limits_mod._imagen_rpm_limiter = None
    yield
    rate_limits_mod._imagen_semaphore = None
    rate_limits_mod._openalex_semaphore = None
    rate_limits_mod._mmdc_semaphore = None
    rate_limits_mod._imagen_daily_tracker = None
    rate_limits_mod._imagen_rpm_limiter = None


# ---------------------------------------------------------------------------
# Semaphore tests (existing)
# ---------------------------------------------------------------------------


class TestGetImagenSemaphore:
    def test_lazy_init_creates_semaphore(self):
        first = get_imagen_semaphore()
        second = get_imagen_semaphore()
        assert isinstance(first, asyncio.Semaphore)
        assert first is second

    def test_default_limit_is_10(self):
        sem = get_imagen_semaphore()
        assert sem._value == 10

    def test_env_var_override(self):
        with patch.dict("os.environ", {"THALA_IMAGEN_CONCURRENCY": "3"}):
            sem = get_imagen_semaphore()
        assert sem._value == 3


class TestGetOpenAlexSemaphore:
    def test_lazy_init_creates_semaphore(self):
        first = get_openalex_semaphore()
        second = get_openalex_semaphore()
        assert isinstance(first, asyncio.Semaphore)
        assert first is second

    def test_default_limit_is_20(self):
        sem = get_openalex_semaphore()
        assert sem._value == 20


class TestSemaphoreIndependence:
    def test_imagen_and_openalex_are_different_instances(self):
        assert get_imagen_semaphore() is not get_openalex_semaphore()


# ---------------------------------------------------------------------------
# ImagenDailyTracker
# ---------------------------------------------------------------------------


class TestImagenDailyTracker:
    @pytest.mark.asyncio
    async def test_try_acquire_succeeds_within_limit(self, tmp_path):
        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=3)
        assert await tracker.try_acquire() is True
        assert await tracker.try_acquire() is True
        assert await tracker.try_acquire() is True

    @pytest.mark.asyncio
    async def test_try_acquire_fails_at_limit(self, tmp_path):
        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=2)
        assert await tracker.try_acquire() is True
        assert await tracker.try_acquire() is True
        assert await tracker.try_acquire() is False

    @pytest.mark.asyncio
    async def test_remaining_reflects_usage(self, tmp_path):
        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=5)
        assert await tracker.remaining() == 5
        await tracker.try_acquire()
        assert await tracker.remaining() == 4
        await tracker.try_acquire()
        assert await tracker.remaining() == 3

    @pytest.mark.asyncio
    async def test_daily_reset_on_date_change(self, tmp_path):
        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=1)

        # Exhaust budget
        assert await tracker.try_acquire() is True
        assert await tracker.try_acquire() is False

        # Simulate date change by writing a stale date
        state_file = tmp_path / "imagen_daily_usage.json"
        state_file.write_text(json.dumps({"date": "2020-01-01", "count": 999}))

        # Should reset and allow new acquisitions
        assert await tracker.try_acquire() is True

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path):
        """Counter persists across tracker instances (simulating process restart)."""
        t1 = ImagenDailyTracker(state_dir=tmp_path, limit=3)
        await t1.try_acquire()
        await t1.try_acquire()

        # New instance reads persisted state
        t2 = ImagenDailyTracker(state_dir=tmp_path, limit=3)
        assert await t2.remaining() == 1
        assert await t2.try_acquire() is True
        assert await t2.try_acquire() is False

    @pytest.mark.asyncio
    async def test_handles_missing_state_file(self, tmp_path):
        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=5)
        # No state file yet — should work fine
        assert await tracker.remaining() == 5

    @pytest.mark.asyncio
    async def test_handles_corrupt_state_file(self, tmp_path):
        state_file = tmp_path / "imagen_daily_usage.json"
        state_file.write_text("not valid json!!!")

        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=5)
        # Should gracefully reset
        assert await tracker.try_acquire() is True

    @pytest.mark.asyncio
    async def test_handles_malformed_state(self, tmp_path):
        state_file = tmp_path / "imagen_daily_usage.json"
        state_file.write_text(json.dumps({"unexpected": "schema"}))

        tracker = ImagenDailyTracker(state_dir=tmp_path, limit=5)
        assert await tracker.try_acquire() is True


# ---------------------------------------------------------------------------
# ImagenRPMLimiter
# ---------------------------------------------------------------------------


class TestImagenRPMLimiter:
    @pytest.mark.asyncio
    async def test_acquire_succeeds_within_burst(self):
        """Token bucket starts full — burst of RPM calls should succeed immediately."""
        limiter = ImagenRPMLimiter(rpm=10)
        # Should be able to acquire up to RPM tokens without waiting
        for _ in range(10):
            await asyncio.wait_for(limiter.acquire(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_exhausted(self):
        """After exhausting tokens, acquire should block (timeout proves it)."""
        limiter = ImagenRPMLimiter(rpm=2)
        await limiter.acquire()
        await limiter.acquire()

        # Third acquire should block since no tokens remain
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(limiter.acquire(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """After waiting, tokens should refill and allow acquisition."""
        limiter = ImagenRPMLimiter(rpm=600)  # 10 per second
        # Exhaust all tokens
        for _ in range(600):
            await limiter.acquire()

        # Wait 0.2s for refill: 0.2 * (600/60) = 2 tokens
        await asyncio.sleep(0.25)
        # Should now be able to acquire
        await asyncio.wait_for(limiter.acquire(), timeout=1.0)


# ---------------------------------------------------------------------------
# Lazy factories
# ---------------------------------------------------------------------------


class TestLazyFactories:
    def test_get_imagen_daily_tracker_returns_singleton(self):
        t1 = get_imagen_daily_tracker()
        t2 = get_imagen_daily_tracker()
        assert t1 is t2

    def test_get_imagen_rpm_limiter_returns_singleton(self):
        l1 = get_imagen_rpm_limiter()
        l2 = get_imagen_rpm_limiter()
        assert l1 is l2

    def test_rpm_limiter_default_is_5(self):
        limiter = get_imagen_rpm_limiter()
        assert limiter._rpm == 5

    def test_rpm_limiter_env_override(self):
        with patch.dict("os.environ", {"THALA_IMAGEN_RPM_LIMIT": "20"}):
            limiter = get_imagen_rpm_limiter()
        assert limiter._rpm == 20


# ---------------------------------------------------------------------------
# reset_rate_limiters
# ---------------------------------------------------------------------------


class TestResetRateLimiters:
    def test_reset_clears_all_globals(self):
        # Create all singletons
        get_imagen_semaphore()
        get_openalex_semaphore()
        get_imagen_daily_tracker()
        get_imagen_rpm_limiter()

        assert rate_limits_mod._imagen_semaphore is not None
        assert rate_limits_mod._openalex_semaphore is not None
        assert rate_limits_mod._imagen_daily_tracker is not None
        assert rate_limits_mod._imagen_rpm_limiter is not None

        reset_rate_limiters()

        assert rate_limits_mod._imagen_semaphore is None
        assert rate_limits_mod._openalex_semaphore is None
        assert rate_limits_mod._imagen_daily_tracker is None
        assert rate_limits_mod._imagen_rpm_limiter is None

    def test_new_instances_after_reset(self):
        old_sem = get_imagen_semaphore()
        old_tracker = get_imagen_daily_tracker()

        reset_rate_limiters()

        new_sem = get_imagen_semaphore()
        new_tracker = get_imagen_daily_tracker()

        assert new_sem is not old_sem
        assert new_tracker is not old_tracker
