"""Tests for task context propagation via ContextVar."""

import asyncio

import pytest

from core.task_queue.task_context import (
    clear_task_context,
    get_task_context,
    get_trace_metadata,
    get_trace_tags,
    set_task_context,
)


@pytest.fixture(autouse=True)
def _clean_context():
    clear_task_context()
    yield
    clear_task_context()


class TestTaskContext:
    """Test the set/get/clear lifecycle."""

    def test_default_is_none(self):
        assert get_task_context() is None

    def test_set_and_get(self):
        set_task_context("42", "lit_review_full", "AI in Healthcare", "standard")
        ctx = get_task_context()
        assert ctx is not None
        assert ctx.task_id == "42"
        assert ctx.task_type == "lit_review_full"
        assert ctx.topic == "AI in Healthcare"
        assert ctx.quality_tier == "standard"

    def test_clear(self):
        set_task_context("1", "web_research", "Topic", "quick")
        clear_task_context()
        assert get_task_context() is None

    def test_context_is_frozen(self):
        set_task_context("1", "web_research", "Topic", "quick")
        ctx = get_task_context()
        with pytest.raises(AttributeError):
            ctx.task_id = "changed"  # type: ignore[misc]


class TestTraceMetadata:
    """Test get_trace_metadata() helper."""

    def test_returns_empty_dict_when_no_context(self):
        assert get_trace_metadata() == {}

    def test_returns_metadata_when_context_set(self):
        set_task_context("99", "lit_review_full", "Deep Learning", "comprehensive")
        meta = get_trace_metadata()
        assert meta == {
            "task_id": "99",
            "task_type": "lit_review_full",
            "topic": "Deep Learning",
            "quality_tier": "comprehensive",
        }

    def test_truncates_long_topic_to_100_chars(self):
        long_topic = "A" * 200
        set_task_context("1", "web_research", long_topic, "quick")
        meta = get_trace_metadata()
        assert len(meta["topic"]) == 100
        assert meta["topic"] == "A" * 100


class TestTraceTags:
    """Test get_trace_tags() helper."""

    def test_returns_empty_list_when_no_context(self):
        assert get_trace_tags() == []

    def test_returns_tags_when_context_set(self):
        set_task_context("7", "web_research", "Climate Change", "standard")
        tags = get_trace_tags()
        assert tags == ["task:7", "type:web_research"]


class TestAsyncIsolation:
    """Verify ContextVar isolation across async tasks."""

    def test_context_does_not_leak_across_tasks(self):
        """Each asyncio.Task should have its own ContextVar copy."""
        results = {}

        async def worker(name: str, task_id: str):
            set_task_context(task_id, "test", name, "quick")
            await asyncio.sleep(0.01)  # yield to event loop
            ctx = get_task_context()
            results[name] = ctx.task_id if ctx else None

        async def main():
            clear_task_context()
            t1 = asyncio.create_task(worker("a", "100"))
            t2 = asyncio.create_task(worker("b", "200"))
            await asyncio.gather(t1, t2)

        asyncio.run(main())
        assert results["a"] == "100"
        assert results["b"] == "200"
