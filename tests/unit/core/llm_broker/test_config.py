"""Unit tests for LLM Broker configuration."""

import os
from unittest.mock import patch


from core.llm_broker.config import (
    BrokerConfig,
    get_broker_config,
    reset_broker_config,
    set_broker_config,
)
from core.llm_broker.schemas import UserMode


class TestBrokerConfig:
    """Tests for BrokerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BrokerConfig()

        assert config.default_mode == UserMode.BALANCED
        assert config.batch_threshold == 50
        assert config.max_queue_size == 100
        assert config.overflow_behavior == "sync"
        assert config.initial_wait_hours == 1.0
        assert config.balanced_retry_hours == 3.0
        assert config.economical_retry_hours == 12.0
        assert config.poll_interval_seconds == 60
        assert config.max_concurrent_sync == 5
        assert config.queue_dir == ".thala/llm_broker"
        assert config.enable_metrics is True

    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        # Clear any existing env vars
        env_vars = [
            "THALA_LLM_BROKER_MODE",
            "THALA_LLM_BROKER_BATCH_THRESHOLD",
            "THALA_LLM_BROKER_MAX_QUEUE",
            "THALA_LLM_BROKER_OVERFLOW",
            "THALA_LLM_BROKER_QUEUE_DIR",
        ]
        with patch.dict(os.environ, {}, clear=True):
            for var in env_vars:
                os.environ.pop(var, None)

            config = BrokerConfig.from_env()

            assert config.default_mode == UserMode.BALANCED
            assert config.batch_threshold == 50
            assert config.max_queue_size == 100
            assert config.overflow_behavior == "sync"

    def test_from_env_custom_values(self):
        """Test from_env with custom environment variables."""
        env = {
            "THALA_LLM_BROKER_MODE": "fast",
            "THALA_LLM_BROKER_BATCH_THRESHOLD": "25",
            "THALA_LLM_BROKER_MAX_QUEUE": "200",
            "THALA_LLM_BROKER_OVERFLOW": "reject",
            "THALA_LLM_BROKER_QUEUE_DIR": "/custom/path",
        }
        with patch.dict(os.environ, env, clear=False):
            config = BrokerConfig.from_env()

            assert config.default_mode == UserMode.FAST
            assert config.batch_threshold == 25
            assert config.max_queue_size == 200
            assert config.overflow_behavior == "reject"
            assert config.queue_dir == "/custom/path"

    def test_from_env_economical_mode(self):
        """Test from_env with economical mode."""
        with patch.dict(os.environ, {"THALA_LLM_BROKER_MODE": "economical"}):
            config = BrokerConfig.from_env()
            assert config.default_mode == UserMode.ECONOMICAL

    def test_from_env_invalid_mode_defaults_to_balanced(self):
        """Test from_env with invalid mode defaults to balanced."""
        with patch.dict(os.environ, {"THALA_LLM_BROKER_MODE": "invalid"}):
            config = BrokerConfig.from_env()
            assert config.default_mode == UserMode.BALANCED

    def test_get_wait_timeout_hours_initial(self):
        """Test get_wait_timeout_hours for initial attempt."""
        config = BrokerConfig()

        # All modes should use initial_wait_hours for retry_count=0
        assert config.get_wait_timeout_hours(UserMode.FAST, 0) == 1.0
        assert config.get_wait_timeout_hours(UserMode.BALANCED, 0) == 1.0
        assert config.get_wait_timeout_hours(UserMode.ECONOMICAL, 0) == 1.0

    def test_get_wait_timeout_hours_balanced_retries(self):
        """Test get_wait_timeout_hours for balanced mode retries."""
        config = BrokerConfig()

        # Balanced mode: retry 1+ uses balanced_retry_hours
        assert config.get_wait_timeout_hours(UserMode.BALANCED, 1) == 3.0
        assert config.get_wait_timeout_hours(UserMode.BALANCED, 2) == 3.0

    def test_get_wait_timeout_hours_economical_retries(self):
        """Test get_wait_timeout_hours for economical mode retries."""
        config = BrokerConfig()

        # Economical mode: retry 1 uses balanced, retry 2+ uses economical
        assert config.get_wait_timeout_hours(UserMode.ECONOMICAL, 1) == 3.0
        assert config.get_wait_timeout_hours(UserMode.ECONOMICAL, 2) == 12.0
        assert config.get_wait_timeout_hours(UserMode.ECONOMICAL, 3) == 12.0

    def test_max_retries_for_mode(self):
        """Test max_retries_for_mode for each mode."""
        config = BrokerConfig()

        assert config.max_retries_for_mode(UserMode.FAST) == 0
        assert config.max_retries_for_mode(UserMode.BALANCED) == 2
        assert config.max_retries_for_mode(UserMode.ECONOMICAL) == 3


class TestGlobalConfig:
    """Tests for global config management functions."""

    def setup_method(self):
        """Reset global config before each test."""
        reset_broker_config()

    def teardown_method(self):
        """Reset global config after each test."""
        reset_broker_config()

    def test_get_broker_config_creates_default(self):
        """Test get_broker_config creates default if not set."""
        config = get_broker_config()
        assert config is not None
        assert config.default_mode == UserMode.BALANCED

    def test_get_broker_config_returns_same_instance(self):
        """Test get_broker_config returns same instance."""
        config1 = get_broker_config()
        config2 = get_broker_config()
        assert config1 is config2

    def test_set_broker_config(self):
        """Test set_broker_config replaces global config."""
        custom_config = BrokerConfig(default_mode=UserMode.FAST)
        set_broker_config(custom_config)

        config = get_broker_config()
        assert config is custom_config
        assert config.default_mode == UserMode.FAST

    def test_reset_broker_config(self):
        """Test reset_broker_config clears global config."""
        # Set custom config
        custom_config = BrokerConfig(batch_threshold=10)
        set_broker_config(custom_config)

        # Reset
        reset_broker_config()

        # Should create new default config
        config = get_broker_config()
        assert config is not custom_config
        assert config.batch_threshold == 50  # Default value
