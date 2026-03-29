"""Tests for the CLI backend (claude -p) routing."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from workflows.shared.llm_utils.cli_backend import (
    _TIER_TO_CLI_MODEL,
    _build_base_cmd,
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
