"""
GitHub Copilot Anthropic Messages transformation config.

Extends AnthropicMessagesConfig so that Copilot Claude models go directly
to Copilot's native /v1/messages endpoint — preserving thinking blocks,
context_management, and other Anthropic-native features that are lost when
translating through chat/completions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GithubCopilotError,
    GetAPIKeyError,
    get_copilot_default_headers,
)


class GithubCopilotAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    GitHub Copilot Anthropic Messages configuration.

    Subclasses AnthropicMessagesConfig to provide Copilot-specific:
    - Authentication via GitHub Copilot OAuth token flow
    - URL routing to Copilot's /v1/messages endpoint
    - Copilot-required headers (editor-version, user-agent, etc.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        Set up authentication and headers for Copilot's /v1/messages endpoint.

        Uses the Copilot Authenticator for dynamic API key acquisition,
        then merges Copilot-specific headers (editor-version, user-agent, etc.)
        with standard Anthropic headers (anthropic-version, content-type).
        """
        # Get Copilot API key and default headers
        try:
            copilot_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            from litellm.exceptions import AuthenticationError

            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=str(e),
            )

        copilot_headers = get_copilot_default_headers(copilot_api_key)

        # Build merged headers: copilot defaults < caller headers
        merged_headers = {**copilot_headers, **headers}

        # Ensure anthropic-version is set
        if "anthropic-version" not in merged_headers:
            merged_headers["anthropic-version"] = "2023-06-01"

        # Auto-inject anthropic-beta headers for context_management, etc.
        merged_headers = self._update_headers_with_anthropic_beta(
            headers=merged_headers,
            optional_params=optional_params,
        )

        # X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        merged_headers["X-Initiator"] = initiator

        # Vision header if request contains images
        if self._has_vision_content(messages):
            merged_headers["Copilot-Vision-Request"] = "true"

        # Resolve API base
        dynamic_api_base = (
            api_base
            or self.authenticator.get_api_base()
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )

        return merged_headers, dynamic_api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Build the Copilot /v1/messages URL.

        Uses api_base from validate_anthropic_messages_environment (which
        already resolved authenticator/default), or falls back again.
        """
        base = (
            api_base
            or self.authenticator.get_api_base()
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )
        base = base.rstrip("/")
        if base.endswith("/v1/messages"):
            return base
        return f"{base}/v1/messages"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> GithubCopilotError:
        return GithubCopilotError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Copilot /v1/messages is stricter than native Anthropic about
        request validation. Apply provider-specific sanitisation:

        1. Copy metadata to break shared dict reference with
           litellm_params (pre_call injects raw_request after
           transform but before json.dumps).
        2. Filter out empty text content blocks — Copilot rejects
           them with "text content blocks must be non-empty".
           Same issue as Databricks (see databricks/chat/transformation.py).
        """
        # 1. Break metadata shared reference
        metadata = anthropic_messages_optional_request_params.get("metadata")
        if isinstance(metadata, dict):
            anthropic_messages_optional_request_params["metadata"] = dict(metadata)

        # 2. Filter empty text content blocks from messages
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if content is None:
                message.pop("content", None)
            elif isinstance(content, str):
                if not content.strip():
                    message.pop("content")
            elif isinstance(content, list):
                if not content:
                    message.pop("content")
                else:
                    filtered = [
                        block for block in content
                        if not (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and not (block.get("text") or "").strip()
                        )
                    ]
                    if not filtered:
                        message.pop("content")
                    else:
                        message["content"] = filtered

        return super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    # -- Helper methods reused from GithubCopilotConfig --

    @staticmethod
    def _determine_initiator(messages: List[Any]) -> str:
        """
        Determine if request is user or agent initiated based on message roles.
        Returns 'agent' if any message has role 'tool' or 'assistant', otherwise 'user'.
        """
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                if role in ("tool", "assistant"):
                    return "agent"
        return "user"

    @staticmethod
    def _has_vision_content(messages: List[Any]) -> bool:
        """
        Check if any message contains vision content (images).
        Checks for image_url content type and type='image_url' items.
        """
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        if "image_url" in content_item:
                            return True
                        if content_item.get("type") == "image_url":
                            return True
        return False
