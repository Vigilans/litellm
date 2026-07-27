"""
Tests for ChatGPT subscription image generation transformation

Source: litellm/llms/chatgpt/image_generation/transformation.py
"""

from unittest.mock import MagicMock

import httpx
import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.image_generation.transformation import (
    ChatGPTImageGenerationConfig,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.fixture
def authenticator():
    auth = MagicMock()
    auth.get_access_token.return_value = "access-123"
    auth.get_account_id.return_value = "acct-123"
    auth.get_api_base.return_value = "https://chatgpt.example.com"
    return auth


@pytest.fixture
def config(authenticator):
    return ChatGPTImageGenerationConfig(authenticator=authenticator)


def _response(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body, headers={"content-type": "application/json"})


class TestChatGPTImageGenerationRegistration:
    def test_provider_config_registration(self):
        config = ProviderConfigManager.get_provider_image_generation_config(
            model="gpt-image-2",
            provider=LlmProviders.CHATGPT,
        )

        assert isinstance(config, ChatGPTImageGenerationConfig)

    def test_cost_map_entry_is_priceable(self):
        """`default_image_cost_calculator` raises unless the entry carries
        `input_cost_per_image` or `input_cost_per_pixel`, so a cost-free
        subscription model still needs an explicit zero."""
        import litellm

        cost_info = litellm.model_cost["chatgpt/gpt-image-2"]

        assert cost_info["mode"] == "image_generation"
        assert cost_info["litellm_provider"] == "chatgpt"
        assert (
            cost_info.get("input_cost_per_image") is not None
            or cost_info.get("input_cost_per_pixel") is not None
        )

    def test_image_generation_cost_is_zero_and_does_not_raise(self):
        """Image generation is billed against the ChatGPT subscription, so the
        proxy must record zero spend rather than fail cost calculation."""
        from litellm.cost_calculator import default_image_cost_calculator

        cost = default_image_cost_calculator(
            model="chatgpt/gpt-image-2",
            custom_llm_provider="chatgpt",
            quality="low",
            n=1,
            size="1254x1254",
        )

        assert cost == 0.0

    def test_images_main_routes_chatgpt_through_provider_config(self):
        """chatgpt is in `litellm.openai_compatible_providers`, so without an
        explicit branch it falls through to the plain OpenAI SDK path, which
        cannot attach the ChatGPT-specific auth headers."""
        import inspect

        import litellm.images.main as images_main

        source = inspect.getsource(images_main.image_generation)
        provider_config_branch = source.index("litellm.LlmProviders.RECRAFT")
        openai_compatible_branch = source.index("litellm.openai_compatible_providers")

        assert "litellm.LlmProviders.CHATGPT" in source
        assert source.index("litellm.LlmProviders.CHATGPT") < openai_compatible_branch
        assert provider_config_branch < openai_compatible_branch


class TestChatGPTImageGenerationURL:
    def test_url_uses_authenticator_api_base(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://chatgpt.example.com/images/generations"

    @pytest.mark.parametrize(
        "api_base",
        ["https://custom.chatgpt.com", "https://custom.chatgpt.com/"],
    )
    def test_url_honors_explicit_api_base(self, config, api_base):
        url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://custom.chatgpt.com/images/generations"


class TestChatGPTImageGenerationEnvironment:
    def test_validate_environment_sets_codex_headers(self, config):
        headers = config.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={"litellm_session_id": "session-123"},
        )

        assert headers["Authorization"] == "Bearer access-123"
        assert headers["ChatGPT-Account-Id"] == "acct-123"
        assert headers["session_id"] == "session-123"
        assert headers["content-type"] == "application/json"
        assert headers["originator"]

    def test_validate_environment_requests_json_not_sse(self, config):
        """The chat/responses default headers ask for `text/event-stream`;
        the image endpoint answers with a plain JSON body."""
        headers = config.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )

        assert headers["accept"] == "application/json"

    def test_validate_environment_sends_non_default_user_agent(self, config):
        """Cloudflare rejects the default httpx user-agent with error 1010."""
        headers = config.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )

        assert "httpx" not in headers["user-agent"]
        assert headers["user-agent"].startswith(headers["originator"])

    def test_caller_headers_take_precedence(self, config):
        headers = config.validate_environment(
            headers={"originator": "custom-origin"},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )

        assert headers["originator"] == "custom-origin"

    def test_login_failure_raises_authentication_error(self, authenticator):
        authenticator.get_access_token.side_effect = GetAccessTokenError(
            status_code=401, message="not logged in"
        )
        config = ChatGPTImageGenerationConfig(authenticator=authenticator)

        with pytest.raises(AuthenticationError) as exc_info:
            config.validate_environment(
                headers={},
                model="gpt-image-2",
                messages=[],
                optional_params={},
                litellm_params={},
            )

        assert "not logged in" in str(exc_info.value)


class TestChatGPTImageGenerationRequest:
    def test_request_body_carries_prompt_and_optional_params(self, config):
        body = config.transform_image_generation_request(
            model="gpt-image-2",
            prompt="a red circle",
            optional_params={"size": "1024x1024", "quality": "low", "n": 1},
            litellm_params={},
            headers={},
        )

        assert body == {
            "model": "gpt-image-2",
            "prompt": "a red circle",
            "size": "1024x1024",
            "quality": "low",
            "n": 1,
        }


class TestChatGPTImageGenerationResponse:
    """Payload shape captured from a live
    POST https://chatgpt.com/backend-api/codex/images/generations call."""

    BODY = {
        "created": 1785083947,
        "background": "opaque",
        "data": [{"b64_json": "aGVsbG8="}],
        "output_format": "png",
        "quality": "low",
        "size": "1254x1254",
        "usage": {
            "input_tokens": 17,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 17},
            "output_tokens": 229,
            "output_tokens_details": {"image_tokens": 229, "text_tokens": 0},
            "total_tokens": 246,
        },
    }

    def _transform(self, config, optional_params) -> ImageResponse:
        return config.transform_image_generation_response(
            model="gpt-image-2",
            raw_response=_response(self.BODY),
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={"prompt": "a red circle"},
            optional_params=optional_params,
            litellm_params={},
            encoding=None,
        )

    def test_image_payload_and_usage_are_preserved(self, config):
        response = self._transform(config, {})

        assert [image.b64_json for image in response.data] == ["aGVsbG8="]
        assert response.created == 1785083947
        assert response.usage.input_tokens == 17
        assert response.usage.output_tokens == 229
        assert response.usage.total_tokens == 246

    def test_reports_server_resolved_size_and_quality(self, config):
        """The backend picks its own size/quality and ignores unsupported
        requested values, so the response must not echo the request."""
        response = self._transform(config, {"size": "auto", "quality": "auto"})

        assert response.size == "1254x1254"
        assert response.quality == "low"
        assert response.output_format == "png"

    def test_ignored_request_size_is_not_reported_back(self, config):
        """A size the backend silently drops must not be reported as applied."""
        response = self._transform(config, {"size": "99x99"})

        assert response.size == "1254x1254"

    def test_usage_is_mapped_for_cost_tracking(self, config):
        response = self._transform(config, {})

        assert response.usage.prompt_tokens == 17
        assert response.usage.completion_tokens == 229
