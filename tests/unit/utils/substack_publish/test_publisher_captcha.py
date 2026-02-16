"""Tests for Substack publisher captcha integration."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from utils.substack_publish.publisher import SubstackPublisher, _extract_substack_site_key


@pytest.fixture
def config():
    return {
        "email": "test@example.com",
        "password": "test-pass",
        "publication_url": "testpub.substack.com",
    }


@pytest.fixture
def publisher(config):
    return SubstackPublisher(config)


class TestLoginWithCaptcha:
    @pytest.mark.asyncio
    async def test_solves_captcha_on_captcha_error(self, publisher):
        """When Api() raises a captcha error, solver is invoked and direct POST used."""
        mock_api = MagicMock()
        mock_api.base_url = "https://substack.com/api"
        mock_api._session.post.return_value = MagicMock(status_code=200)

        with (
            patch("utils.substack_publish.publisher.Api") as MockApi,
            patch(
                "core.captcha.solver.CaptchaSolver.solve_recaptcha_v2",
                new_callable=AsyncMock,
                return_value="solved-token",
            ),
            patch(
                "utils.substack_publish.publisher._extract_substack_site_key",
                return_value=None,
            ),
        ):
            # First call raises captcha error, second creates unauthenticated Api
            MockApi.side_effect = [Exception("captcha required"), mock_api]
            api = await publisher._create_api()

        assert api is mock_api
        # Verify the direct POST was made with the captcha token
        call_args = mock_api._session.post.call_args
        assert call_args[1]["json"]["captcha_response"] == "solved-token"
        assert call_args[1]["json"]["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_falls_back_to_cookies_when_solver_fails(self, publisher, tmp_path):
        """When captcha solve fails, falls back to cookie auth."""
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("{}")
        publisher.config["cookies_path"] = str(cookie_file)

        mock_cookie_api = MagicMock()

        with (
            patch("utils.substack_publish.publisher.Api") as MockApi,
            patch(
                "core.captcha.solver.CaptchaSolver.solve_recaptcha_v2",
                new_callable=AsyncMock,
                side_effect=Exception("Solve failed"),
            ),
            patch(
                "utils.substack_publish.publisher._extract_substack_site_key",
                return_value=None,
            ),
        ):
            # First call raises captcha, second (cookie) succeeds
            MockApi.side_effect = [
                Exception("captcha required"),
                mock_cookie_api,
            ]
            api = await publisher._create_api()

        # Should have fallen back to cookie auth
        assert api is mock_cookie_api

    @pytest.mark.asyncio
    async def test_skips_captcha_when_no_api_key(self, publisher, tmp_path):
        """When CAPSOLVER_API_KEY is not set, captcha solving is skipped."""
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("{}")
        publisher.config["cookies_path"] = str(cookie_file)

        mock_cookie_api = MagicMock()

        with (
            patch("utils.substack_publish.publisher.Api") as MockApi,
            patch.dict("os.environ", {}, clear=False),
            patch(
                "core.captcha.config.os.environ.get",
                side_effect=lambda k, d=None: None if k == "CAPSOLVER_API_KEY" else d,
            ),
        ):
            MockApi.side_effect = [
                Exception("captcha required"),
                mock_cookie_api,
            ]
            api = await publisher._create_api()

        assert api is mock_cookie_api

    @pytest.mark.asyncio
    async def test_non_captcha_error_falls_through(self, publisher, tmp_path):
        """Non-captcha auth errors fall through without attempting solve."""
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text("{}")
        publisher.config["cookies_path"] = str(cookie_file)

        mock_cookie_api = MagicMock()

        with (
            patch("utils.substack_publish.publisher.Api") as MockApi,
            patch(
                "core.captcha.solver.CaptchaSolver.solve_recaptcha_v2",
                new_callable=AsyncMock,
            ) as mock_solve,
        ):
            MockApi.side_effect = [
                Exception("invalid credentials"),
                mock_cookie_api,
            ]
            api = await publisher._create_api()

        # Solver should NOT have been called for non-captcha errors
        mock_solve.assert_not_called()
        assert api is mock_cookie_api


class TestExtractSubstackSiteKey:
    def test_extracts_from_preloads(self):
        html = '<script>window._preloads = {"captcha_site_key": "6Le-abc123"}</script>'
        with patch("utils.substack_publish.publisher.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.text = html
            mock_resp.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_resp
            key = _extract_substack_site_key()
        assert key == "6Le-abc123"

    def test_extracts_from_data_sitekey(self):
        html = '<div class="g-recaptcha" data-sitekey="6Le-xyz789"></div>'
        with patch("utils.substack_publish.publisher.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.text = html
            mock_resp.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_resp
            key = _extract_substack_site_key()
        assert key == "6Le-xyz789"

    def test_returns_none_on_http_error(self):
        with patch("utils.substack_publish.publisher.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("Network error")
            key = _extract_substack_site_key()
        assert key is None

    def test_returns_none_when_no_key_found(self):
        with patch("utils.substack_publish.publisher.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.text = "<html><body>No captcha here</body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_httpx.get.return_value = mock_resp
            key = _extract_substack_site_key()
        assert key is None
