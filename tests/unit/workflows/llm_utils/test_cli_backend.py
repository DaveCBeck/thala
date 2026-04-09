"""Tests for the CLI backend (claude -p) routing."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from workflows.shared.llm_utils.cli_backend import (
    _TIER_TO_CLI_MODEL,
    _RateLimitError,
    _build_base_cmd,
    _check_rate_limit,
    _run_claude_cli,
    invoke_structured_via_cli,
    invoke_via_cli,
    is_cli_backend_enabled,
)
from workflows.shared.llm_utils.config import InvokeConfig
from workflows.shared.llm_utils.invoke import invoke
from workflows.shared.llm_utils.models import ModelTier


# _run_claude_cli calls os.waitpid to reap the subprocess synchronously.
# Mock it out for all tests that spawn a mock subprocess.
@pytest.fixture(autouse=True)
def _mock_waitpid():
    with patch("workflows.shared.llm_utils.cli_backend.os.waitpid", return_value=(0, 0)):
        yield


# -- Test schema for structured output --


class MockAnalysis(BaseModel):
    summary: str
    score: float


# -- Helpers --


def _make_proc_mock(stdout_data: dict, returncode: int = 0, stderr: str = ""):
    """Create a mock subprocess result with pipe-based I/O.

    Also patches os.waitpid to return the matching exit status so
    _run_claude_cli sees the correct return code.
    """
    proc = AsyncMock()
    proc.pid = 99999
    proc.returncode = returncode

    # Mock stdin (write + drain + close)
    proc.stdin = AsyncMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()

    # Mock stdout/stderr readers
    proc.stdout = AsyncMock()
    proc.stdout.read = AsyncMock(return_value=json.dumps(stdout_data).encode())
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=stderr.encode())

    # waitpid status: encode returncode as a normal exit status (rc << 8)
    proc._waitpid_status = returncode << 8
    return proc


def _text_envelope(text: str) -> dict:
    return {"type": "result", "subtype": "success", "result": text}


def _structured_envelope(data: dict) -> dict:
    return {"type": "result", "subtype": "success", "result": "", "structured_output": data}


# -- _check_rate_limit parser --


class TestCheckRateLimit:
    def test_empty_is_noop(self):
        _check_rate_limit("")
        _check_rate_limit(None)  # type: ignore[arg-type]

    def test_no_marker_is_noop(self):
        _check_rate_limit("normal output from claude")

    def test_epoch_form(self):
        with pytest.raises(_RateLimitError) as exc:
            _check_rate_limit("Claude AI usage limit reached|1774177200")
        assert exc.value.reset_epoch == 1774177200.0

    def test_epoch_with_limit_type(self):
        """Newer format includes the limit type after a second pipe."""
        with pytest.raises(_RateLimitError) as exc:
            _check_rate_limit("Claude AI usage limit reached|1774177200|weekly")
        assert exc.value.reset_epoch == 1774177200.0

    def test_marker_embedded_in_larger_text(self):
        """The result field may wrap the marker in prose."""
        payload = 'Error: "Claude AI usage limit reached|1774177200" please retry later'
        with pytest.raises(_RateLimitError) as exc:
            _check_rate_limit(payload)
        assert exc.value.reset_epoch == 1774177200.0

    def test_plain_fallback_no_epoch(self):
        with pytest.raises(_RateLimitError) as exc:
            _check_rate_limit("Claude usage limit reached. Your limit will reset later.")
        assert exc.value.reset_epoch is None


# -- is_cli_backend_enabled --


class TestIsCliBackendEnabled:
    def test_default_is_api(self):
        with patch.dict("os.environ", {}, clear=True):
            assert is_cli_backend_enabled() is False

    def test_api_explicit(self):
        with patch.dict("os.environ", {"THALA_LLM_BACKEND": "api"}):
            assert is_cli_backend_enabled() is False

    def test_cli_enabled(self):
        with patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}):
            assert is_cli_backend_enabled() is True

    def test_cli_case_insensitive(self):
        with patch.dict("os.environ", {"THALA_LLM_BACKEND": "CLI"}):
            assert is_cli_backend_enabled() is True


# -- _build_base_cmd --


class TestBuildBaseCmd:
    def test_basic_command(self):
        cmd = _build_base_cmd("sonnet", "You are helpful.")
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert "--system-prompt" in cmd
        assert "--tools" in cmd
        assert "" in cmd  # tools disabled
        assert "--max-turns" in cmd
        assert "--no-session-persistence" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--effort" not in cmd

    def test_with_effort(self):
        cmd = _build_base_cmd("opus", "System.", effort="high")
        idx = cmd.index("--effort")
        assert cmd[idx + 1] == "high"


# -- invoke_via_cli --


class TestInvokeViaCli:
    @pytest.mark.asyncio
    async def test_text_response(self):
        envelope = _text_envelope("Hello world")
        proc = _make_proc_mock(envelope)

        with patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc):
            result = await invoke_via_cli(ModelTier.SONNET, "Be helpful.", "Say hello")

        assert isinstance(result, AIMessage)
        assert result.content == "Hello world"

    @pytest.mark.asyncio
    async def test_with_effort(self):
        envelope = _text_envelope("Thought deeply")
        proc = _make_proc_mock(envelope)

        with patch(
            "workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc
        ) as mock_exec:
            await invoke_via_cli(ModelTier.OPUS, "System.", "Query", effort="high")

        # Verify --effort was passed
        call_args = mock_exec.call_args[0]
        assert "--effort" in call_args
        assert "high" in call_args

    @pytest.mark.asyncio
    async def test_subprocess_failure_raises(self):
        proc = _make_proc_mock({}, returncode=1, stderr="Something broke")

        with (
            patch("workflows.shared.llm_utils.cli_backend._CLI_MAX_RETRIES", 1),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc),
            patch("workflows.shared.llm_utils.cli_backend.os.waitpid", return_value=(99999, 1 << 8)),
        ):
            with pytest.raises(RuntimeError, match="claude -p failed"):
                await invoke_via_cli(ModelTier.SONNET, "System.", "Query")

    @pytest.mark.asyncio
    async def test_model_mapping(self):
        """Each Claude tier maps to the correct --model value."""
        for tier, expected_model in _TIER_TO_CLI_MODEL.items():
            envelope = _text_envelope("ok")
            proc = _make_proc_mock(envelope)

            with patch(
                "workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc
            ) as mock_exec:
                await invoke_via_cli(tier, "Sys.", "Query")

            call_args = mock_exec.call_args[0]
            model_idx = list(call_args).index("--model")
            assert call_args[model_idx + 1] == expected_model


# -- invoke_structured_via_cli --


class TestInvokeStructuredViaCli:
    @pytest.mark.asyncio
    async def test_structured_response(self):
        envelope = _structured_envelope({"summary": "Great paper", "score": 0.95})
        proc = _make_proc_mock(envelope)

        with patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc):
            result = await invoke_structured_via_cli(ModelTier.SONNET, "Analyze.", "Paper text", MockAnalysis)

        assert isinstance(result, MockAnalysis)
        assert result.summary == "Great paper"
        assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_json_schema_passed(self):
        envelope = _structured_envelope({"summary": "ok", "score": 0.5})
        proc = _make_proc_mock(envelope)

        with patch(
            "workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc
        ) as mock_exec:
            await invoke_structured_via_cli(ModelTier.SONNET, "Sys.", "Query", MockAnalysis)

        call_args = mock_exec.call_args[0]
        assert "--json-schema" in call_args
        schema_idx = list(call_args).index("--json-schema")
        schema_str = call_args[schema_idx + 1]
        parsed = json.loads(schema_str)
        assert "properties" in parsed
        assert "summary" in parsed["properties"]

    @pytest.mark.asyncio
    async def test_subprocess_failure_raises(self):
        proc = _make_proc_mock({}, returncode=1, stderr="Schema error")

        with (
            patch("workflows.shared.llm_utils.cli_backend._CLI_MAX_RETRIES", 1),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc),
            patch("workflows.shared.llm_utils.cli_backend.os.waitpid", return_value=(99999, 1 << 8)),
        ):
            with pytest.raises(RuntimeError, match="claude -p failed"):
                await invoke_structured_via_cli(ModelTier.SONNET, "Sys.", "Query", MockAnalysis)


# -- invoke() routing with CLI backend --


class TestInvokeCliRouting:
    """Test that invoke() routes correctly when CLI backend is enabled."""

    @pytest.mark.asyncio
    async def test_cli_routes_claude_text(self):
        """Claude text calls should route through CLI when enabled."""
        envelope = _text_envelope("CLI response")
        proc = _make_proc_mock(envelope)

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await invoke(tier=ModelTier.SONNET, system="Sys.", user="Hello")

        assert isinstance(result, AIMessage)
        assert result.content == "CLI response"

    @pytest.mark.asyncio
    async def test_cli_routes_claude_structured(self):
        """Claude structured calls should route through CLI when enabled."""
        envelope = _structured_envelope({"summary": "good", "score": 0.8})
        proc = _make_proc_mock(envelope)

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc),
        ):
            result = await invoke(
                tier=ModelTier.SONNET,
                system="Analyze.",
                user="Content",
                schema=MockAnalysis,
            )

        assert isinstance(result, MockAnalysis)
        assert result.summary == "good"

    @pytest.mark.asyncio
    async def test_cli_falls_through_for_deepseek(self):
        """DeepSeek calls should fall through to API path even when CLI enabled."""
        mock_direct = AsyncMock(return_value=[AIMessage(content="deepseek response")])

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}),
            patch("workflows.shared.llm_utils.invoke._invoke_direct", mock_direct),
        ):
            result = await invoke(tier=ModelTier.DEEPSEEK_V3, system="Sys.", user="Hello")

        assert result.content == "deepseek response"
        mock_direct.assert_called_once()

    @pytest.mark.asyncio
    async def test_cli_falls_through_for_tools(self):
        """Tool-agent calls should fall through to API path even when CLI enabled."""
        mock_structured = AsyncMock(return_value=MockAnalysis(summary="api", score=0.5))

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}),
            patch("workflows.shared.llm_utils.invoke._invoke_structured", mock_structured),
        ):
            result = await invoke(
                tier=ModelTier.SONNET,
                system="Sys.",
                user="Hello",
                schema=MockAnalysis,
                config=InvokeConfig(tools=[MagicMock()]),
            )

        assert isinstance(result, MockAnalysis)
        mock_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_cli_falls_through_for_multimodal(self):
        """Multimodal calls should fall through to API path even when CLI enabled."""
        mock_direct = AsyncMock(return_value=[AIMessage(content="vision response")])

        multimodal_content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image", "source": {"type": "base64", "data": "abc123"}},
        ]

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}),
            patch("workflows.shared.llm_utils.invoke._invoke_direct", mock_direct),
        ):
            result = await invoke(tier=ModelTier.SONNET, system="Sys.", user=multimodal_content)

        assert result.content == "vision response"
        mock_direct.assert_called_once()

    @pytest.mark.asyncio
    async def test_cli_handles_batch_input(self):
        """Batch input should be unwrapped to sequential CLI calls."""
        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_proc_mock(_text_envelope(f"response {call_count}"))

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "cli"}),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", side_effect=mock_subprocess),
        ):
            results = await invoke(
                tier=ModelTier.SONNET,
                system="Sys.",
                user=["prompt 1", "prompt 2", "prompt 3"],
            )

        assert len(results) == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_cli_timeout_raises(self):
        """Subprocess hanging beyond timeout should raise TimeoutError."""
        proc = AsyncMock()
        proc.pid = 99999
        proc.kill = MagicMock()

        # stdin works fine
        proc.stdin = AsyncMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        proc.stdin.close = MagicMock()

        # stdout.read never returns — simulates pipe held open by orphan child
        async def hang_forever():
            await asyncio.sleep(999)

        proc.stdout = AsyncMock()
        proc.stdout.read = hang_forever
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")

        with (
            patch("workflows.shared.llm_utils.cli_backend._CLI_TIMEOUT_SECONDS", 0.1),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc),
            patch("workflows.shared.llm_utils.cli_backend.os.killpg") as mock_killpg,
        ):
            with pytest.raises(TimeoutError, match="timed out"):
                await _run_claude_cli(["claude", "-p"], "prompt")

        mock_killpg.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_in_result_field_sleeps_then_retries(self):
        """When a subscription limit is reported, we sleep until reset and retry."""
        # First call: envelope.result contains the usage-limit marker with an
        # epoch 120 seconds in the future. Second call: clean success.
        reset_epoch = 1_800_000_000
        limit_proc = _make_proc_mock(
            _text_envelope(f"Claude AI usage limit reached|{reset_epoch}|5h")
        )
        success_proc = _make_proc_mock(_text_envelope("after sleep"))

        call_count = 0

        async def fake_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return limit_proc if call_count == 1 else success_proc

        sleep_calls: list[float] = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)

        with (
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch("workflows.shared.llm_utils.cli_backend.time.time", return_value=reset_epoch - 120),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.sleep", side_effect=fake_sleep),
        ):
            result = await invoke_via_cli(ModelTier.SONNET, "Sys.", "Query")

        assert result.content == "after sleep"
        assert call_count == 2
        # Should have slept roughly 120s + 30s buffer (rate-limit sleep).
        # No transient-retry backoff should have fired.
        assert any(140 <= s <= 160 for s in sleep_calls), f"expected ~150s sleep, got {sleep_calls}"

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_consume_retry_budget(self):
        """Rate-limit waits must not count against _CLI_MAX_RETRIES."""
        reset_epoch = 1_800_000_000
        call_count = 0

        async def fake_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # 5 consecutive rate-limits then success — more than _CLI_MAX_RETRIES (4).
            if call_count <= 5:
                return _make_proc_mock(
                    _text_envelope(f"Claude AI usage limit reached|{reset_epoch}")
                )
            return _make_proc_mock(_text_envelope("finally ok"))

        with (
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch("workflows.shared.llm_utils.cli_backend.time.time", return_value=reset_epoch - 1),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.sleep", new=AsyncMock()),
        ):
            result = await invoke_via_cli(ModelTier.SONNET, "Sys.", "Query")

        assert result.content == "finally ok"
        assert call_count == 6

    @pytest.mark.asyncio
    async def test_rate_limit_in_stderr_with_rc1_is_caught(self):
        """Rate-limit detected in stderr (rc!=0) should trigger sleep, not RuntimeError."""
        reset_epoch = 1_800_000_000
        limit_proc = _make_proc_mock(
            {},  # stdout empty/invalid
            returncode=1,
            stderr=f"Claude AI usage limit reached|{reset_epoch}",
        )
        success_proc = _make_proc_mock(_text_envelope("post-limit"))

        call_count = 0

        async def fake_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return limit_proc if call_count == 1 else success_proc

        # waitpid must report rc=1 on first call, rc=0 on second.
        waitpid_calls = 0

        def fake_waitpid(pid, opts):
            nonlocal waitpid_calls
            waitpid_calls += 1
            return (pid, 1 << 8) if waitpid_calls == 1 else (pid, 0)

        with (
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch("workflows.shared.llm_utils.cli_backend.os.waitpid", side_effect=fake_waitpid),
            patch("workflows.shared.llm_utils.cli_backend.time.time", return_value=reset_epoch - 1),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.sleep", new=AsyncMock()),
        ):
            result = await invoke_via_cli(ModelTier.SONNET, "Sys.", "Query")

        assert result.content == "post-limit"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_wait_capped_raises(self):
        """If reset is further out than the cap, we raise instead of sleeping forever."""
        # Reset 10 days in the future, default cap is 6h.
        reset_epoch = 1_800_000_000 + (10 * 86400)
        proc = _make_proc_mock(
            _text_envelope(f"Claude AI usage limit reached|{reset_epoch}|weekly")
        )

        with (
            patch("workflows.shared.llm_utils.cli_backend._CLI_MAX_RETRIES", 1),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.create_subprocess_exec", return_value=proc),
            patch("workflows.shared.llm_utils.cli_backend.time.time", return_value=1_800_000_000),
            patch("workflows.shared.llm_utils.cli_backend.asyncio.sleep", new=AsyncMock()),
        ):
            with pytest.raises(_RateLimitError):
                await invoke_via_cli(ModelTier.SONNET, "Sys.", "Query")

    @pytest.mark.asyncio
    async def test_api_backend_unchanged(self):
        """When backend=api (default), existing routing is used."""
        mock_direct = AsyncMock(return_value=[AIMessage(content="api response")])

        with (
            patch.dict("os.environ", {"THALA_LLM_BACKEND": "api"}),
            patch("workflows.shared.llm_utils.invoke._invoke_direct", mock_direct),
        ):
            result = await invoke(tier=ModelTier.SONNET, system="Sys.", user="Hello")

        assert result.content == "api response"
        mock_direct.assert_called_once()
