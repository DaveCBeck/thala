"""Unit tests for LLM Broker schemas."""

from datetime import datetime, timezone


from core.llm_broker.schemas import (
    BatchPolicy,
    LLMRequest,
    LLMResponse,
    RequestState,
    UserMode,
)


class TestBatchPolicy:
    """Tests for BatchPolicy enum."""

    def test_values(self):
        """Verify all expected policy values exist."""
        assert BatchPolicy.FORCE_BATCH.value == "force_batch"
        assert BatchPolicy.PREFER_BALANCE.value == "prefer_balance"
        assert BatchPolicy.PREFER_SPEED.value == "prefer_speed"
        assert BatchPolicy.REQUIRE_SYNC.value == "sync"


class TestUserMode:
    """Tests for UserMode enum."""

    def test_values(self):
        """Verify all expected mode values exist."""
        assert UserMode.FAST.value == "fast"
        assert UserMode.BALANCED.value == "balanced"
        assert UserMode.ECONOMICAL.value == "economical"


class TestRequestState:
    """Tests for RequestState enum."""

    def test_values(self):
        """Verify all expected state values exist."""
        assert RequestState.QUEUED.value == "queued"
        assert RequestState.SUBMITTED.value == "submitted"
        assert RequestState.COMPLETED.value == "completed"
        assert RequestState.FAILED.value == "failed"


class TestLLMRequest:
    """Tests for LLMRequest dataclass."""

    def test_create_with_defaults(self):
        """Test creating a request with auto-generated ID."""
        request = LLMRequest.create(
            prompt="Test prompt",
            model="claude-sonnet-4-5-20250929",
        )

        assert request.request_id  # Should be non-empty UUID
        assert request.prompt == "Test prompt"
        assert request.model == "claude-sonnet-4-5-20250929"
        assert request.policy == BatchPolicy.PREFER_SPEED
        assert request.state == RequestState.QUEUED
        assert request.max_tokens == 4096
        assert request.system is None
        assert request.effort is None
        assert request.tools is None
        assert request.tool_choice is None
        assert request.batch_id is None
        assert request.retry_count == 0
        assert isinstance(request.created_at, datetime)

    def test_create_with_all_options(self):
        """Test creating a request with all options specified."""
        tools = [{"type": "function", "name": "test"}]
        tool_choice = {"type": "tool", "name": "test"}
        metadata = {"key": "value"}

        request = LLMRequest.create(
            prompt="Test prompt",
            model="claude-opus-4-5-20251101",
            policy=BatchPolicy.FORCE_BATCH,
            max_tokens=8192,
            system="System prompt",
            effort="medium",
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )

        assert request.policy == BatchPolicy.FORCE_BATCH
        assert request.max_tokens == 8192
        assert request.system == "System prompt"
        assert request.effort == "medium"
        assert request.tools == tools
        assert request.tool_choice == tool_choice
        assert request.metadata == metadata

    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict preserve all fields."""
        original = LLMRequest.create(
            prompt="Test prompt",
            model="claude-sonnet-4-5-20250929",
            policy=BatchPolicy.PREFER_BALANCE,
            max_tokens=2048,
            system="System",
            tools=[{"type": "function"}],
            metadata={"foo": "bar"},
        )
        original.state = RequestState.SUBMITTED
        original.batch_id = "batch_123"
        original.submitted_at = datetime.now(timezone.utc)
        original.retry_count = 2

        # Serialize and deserialize
        data = original.to_dict()
        restored = LLMRequest.from_dict(data)

        assert restored.request_id == original.request_id
        assert restored.prompt == original.prompt
        assert restored.model == original.model
        assert restored.policy == original.policy
        assert restored.state == original.state
        assert restored.max_tokens == original.max_tokens
        assert restored.system == original.system
        assert restored.tools == original.tools
        assert restored.batch_id == original.batch_id
        assert restored.retry_count == original.retry_count
        assert restored.metadata == original.metadata

    def test_to_dict_format(self):
        """Test to_dict produces expected format."""
        request = LLMRequest.create(
            prompt="Test",
            model="claude-sonnet-4-5-20250929",
        )

        data = request.to_dict()

        assert "request_id" in data
        assert "prompt" in data
        assert "model" in data
        assert "policy" in data
        assert data["policy"] == "prefer_speed"  # Enum value serialized
        assert "state" in data
        assert data["state"] == "queued"  # Enum value serialized
        assert "created_at" in data
        assert isinstance(data["created_at"], str)  # ISO format string

    def test_from_dict_missing_required_fields(self):
        """Test from_dict raises clear error for missing required fields."""
        import pytest

        # Missing all required fields
        with pytest.raises(ValueError) as exc_info:
            LLMRequest.from_dict({})
        error_msg = str(exc_info.value)
        assert "Missing required fields in queue data:" in error_msg
        assert "request_id" in error_msg
        assert "prompt" in error_msg
        assert "model" in error_msg
        assert "policy" in error_msg
        assert "state" in error_msg

        # Missing some required fields
        with pytest.raises(ValueError) as exc_info:
            LLMRequest.from_dict({"request_id": "123", "prompt": "test"})
        error_msg = str(exc_info.value)
        assert "Missing required fields in queue data:" in error_msg
        assert "model" in error_msg
        assert "policy" in error_msg
        assert "state" in error_msg
        assert "request_id" not in error_msg.split(":")[-1]  # request_id should not be in missing list
        assert "prompt" not in error_msg.split(":")[-1]  # prompt should not be in missing list

    def test_from_dict_invalid_enum_value(self):
        """Test from_dict wraps invalid enum values in descriptive error."""
        import pytest

        data = {
            "request_id": "123",
            "prompt": "test",
            "model": "claude-sonnet-4-5-20250929",
            "policy": "invalid_policy",  # Invalid BatchPolicy value
            "state": "queued",
            "created_at": "2025-01-01T00:00:00+00:00",
        }

        with pytest.raises(ValueError) as exc_info:
            LLMRequest.from_dict(data)
        error_msg = str(exc_info.value)
        assert "Invalid queue data:" in error_msg
        assert exc_info.value.__cause__ is not None  # Original error preserved

    def test_from_dict_invalid_state_enum(self):
        """Test from_dict wraps invalid state enum values in descriptive error."""
        import pytest

        data = {
            "request_id": "123",
            "prompt": "test",
            "model": "claude-sonnet-4-5-20250929",
            "policy": "prefer_speed",
            "state": "invalid_state",  # Invalid RequestState value
            "created_at": "2025-01-01T00:00:00+00:00",
        }

        with pytest.raises(ValueError) as exc_info:
            LLMRequest.from_dict(data)
        error_msg = str(exc_info.value)
        assert "Invalid queue data:" in error_msg
        assert exc_info.value.__cause__ is not None  # Original error preserved


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_successful_response(self):
        """Test creating a successful response."""
        response = LLMResponse(
            request_id="req_123",
            content="Response text",
            success=True,
            usage={"input_tokens": 100, "output_tokens": 50},
            model="claude-sonnet-4-5-20250929",
            stop_reason="end_turn",
        )

        assert response.request_id == "req_123"
        assert response.content == "Response text"
        assert response.success is True
        assert response.error is None
        assert response.usage == {"input_tokens": 100, "output_tokens": 50}
        assert response.model == "claude-sonnet-4-5-20250929"
        assert response.stop_reason == "end_turn"

    def test_failed_response(self):
        """Test creating a failed response."""
        response = LLMResponse(
            request_id="req_456",
            content=None,
            success=False,
            error="Rate limit exceeded",
        )

        assert response.request_id == "req_456"
        assert response.content is None
        assert response.success is False
        assert response.error == "Rate limit exceeded"

    def test_to_dict(self):
        """Test to_dict produces expected format."""
        response = LLMResponse(
            request_id="req_789",
            content="Test content",
            success=True,
            usage={"input_tokens": 10},
        )

        data = response.to_dict()

        assert data["request_id"] == "req_789"
        assert data["content"] == "Test content"
        assert data["success"] is True
        assert data["error"] is None
        assert data["usage"] == {"input_tokens": 10}
