"""Tests for per-publication sequential ID counter."""

from core.task_queue.pub_counter import next_id, peek_id


class TestPubCounter:
    def test_next_id_starts_at_one(self, tmp_path):
        counters_path = tmp_path / "pub_counters.json"
        assert next_id("testpub", counters_path=counters_path) == 1

    def test_next_id_increments(self, tmp_path):
        counters_path = tmp_path / "pub_counters.json"
        assert next_id("testpub", counters_path=counters_path) == 1
        assert next_id("testpub", counters_path=counters_path) == 2
        assert next_id("testpub", counters_path=counters_path) == 3

    def test_next_id_independent_per_pub(self, tmp_path):
        counters_path = tmp_path / "pub_counters.json"
        assert next_id("pub_a", counters_path=counters_path) == 1
        assert next_id("pub_b", counters_path=counters_path) == 1
        assert next_id("pub_a", counters_path=counters_path) == 2

    def test_peek_id_does_not_increment(self, tmp_path):
        counters_path = tmp_path / "pub_counters.json"
        assert peek_id("testpub", counters_path=counters_path) == 1
        assert peek_id("testpub", counters_path=counters_path) == 1
        # Now actually claim it
        assert next_id("testpub", counters_path=counters_path) == 1
        assert peek_id("testpub", counters_path=counters_path) == 2
