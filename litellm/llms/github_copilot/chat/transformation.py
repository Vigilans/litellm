from typing import List, Optional, Tuple

import os

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAPIKeyError,
    get_copilot_default_headers,
)


class GithubCopilotConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = (
            api_base
            or self.authenticator.get_api_base()
            or os.getenv("GITHUB_COPILOT_API_BASE")
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )
        try:
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def _transform_messages(
        self,
        messages,
        model: str,
    ):
        import litellm

        # Check if system-to-assistant conversion is disabled
        if litellm.disable_copilot_system_to_assistant:
            # GitHub Copilot API now supports system prompts for all models (Claude, GPT, etc.)
            # No conversion needed - just return messages as-is
            return messages

        # Default behavior: convert system messages to assistant for compatibility
        transformed_messages = []
        for message in messages:
            if message.get("role") == "system":
                # Convert system message to assistant message
                transformed_message = message.copy()
                transformed_message["role"] = "assistant"
                transformed_messages.append(transformed_message)
            else:
                transformed_messages.append(message)

        return transformed_messages

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Get base headers from parent
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        # Add Copilot-specific headers (editor-version, user-agent, etc.)
        try:
            copilot_api_key = self.authenticator.get_api_key()
            copilot_headers = get_copilot_default_headers(copilot_api_key)
            validated_headers = {**copilot_headers, **validated_headers}
        except GetAPIKeyError:
            pass  # Will be handled later in the request flow

        # Add X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        validated_headers["X-Initiator"] = initiator

        # Add Copilot-Vision-Request header if request contains images
        if self._has_vision_content(messages):
            validated_headers["Copilot-Vision-Request"] = "true"

        return validated_headers

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        result = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # ── Convert thinking → reasoning_effort (all models) ──
        # Copilot's chat/completions only recognises reasoning_effort — even for
        # Claude models. Convert thinking → reasoning_effort for all families.
        thinking = result.pop("thinking", None) or non_default_params.get("thinking")
        if thinking is not None and "reasoning_effort" not in result:
            from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
                LiteLLMAnthropicMessagesAdapter,
            )
            from litellm.llms.anthropic.experimental_pass_through.utils import (
                is_reasoning_auto_summary_enabled,
            )

            effort = LiteLLMAnthropicMessagesAdapter.translate_anthropic_thinking_to_reasoning_effort(thinking)
            if effort is not None:
                summary = thinking.get("summary") if isinstance(thinking, dict) else None
                auto_summary = is_reasoning_auto_summary_enabled()
                if summary:
                    result["reasoning_effort"] = {"effort": effort, "summary": summary}
                elif auto_summary:
                    result["reasoning_effort"] = {"effort": effort, "summary": "detailed"}
                else:
                    result["reasoning_effort"] = effort

        # ── Model-family adjustments ──
        if "claude" in model.lower():
            # Copilot Claude rejects dict reasoning_effort ({"effort":…,"summary":…})
            # with 400 "got object, want string". Flatten to plain string.
            if "reasoning_effort" in result and isinstance(result["reasoning_effort"], dict):
                result["reasoning_effort"] = result["reasoning_effort"].get("effort", "high")

            # context_management: move into extra_body for Claude models
            # (Copilot Claude uses Anthropic-format context_management)
            extra_body = result.get("extra_body", {}) or {}
            cm = result.pop("context_management", None) or non_default_params.get("context_management")
            if cm is not None:
                extra_body["context_management"] = cm
            if extra_body:
                result["extra_body"] = extra_body

        elif model.lower().startswith(("gpt-", "o1", "o3", "o4")):
            # super() may flatten dict reasoning_effort to a plain string,
            # losing the "summary" field. Restore the original dict if needed.
            if "reasoning_effort" in result:
                original = non_default_params.get("reasoning_effort")
                if isinstance(original, dict) and isinstance(result["reasoning_effort"], str):
                    result["reasoning_effort"] = original

            # context_management: convert Anthropic format → OpenAI format if needed
            # Anthropic: {"edits": [{"type": "compact_20260112", "trigger": {...}}]}
            # OpenAI:   [{"type": "compaction", "compact_threshold": 200000}]
            cm = result.pop("context_management", None) or non_default_params.get("context_management")
            if cm is not None:
                if isinstance(cm, dict) and "edits" in cm:
                    from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
                        LiteLLMAnthropicToResponsesAPIAdapter,
                    )
                    converted = LiteLLMAnthropicToResponsesAPIAdapter.translate_context_management_to_responses_api(cm)
                    if converted is not None:
                        result["context_management"] = converted
                elif isinstance(cm, list):
                    # Already in OpenAI format — pass through
                    result["context_management"] = cm

        # ── Universal constraints ──
        # Clamp max_tokens / max_output_tokens to Copilot's minimum (16).
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
            if key in result and isinstance(result[key], int) and result[key] < 16:
                result[key] = 16

        return result

    @staticmethod
    def _normalize_claude_model_name(model: str) -> str:
        """
        Normalize Claude model names from Copilot's dot notation to litellm's
        hyphen notation for capability lookups.

        e.g. "claude-sonnet-4.6" -> "claude-sonnet-4-6"
             "claude-opus-4.6" -> "claude-opus-4-6"
             "claude-opus-4.6-xxxx" -> "claude-opus-4-6"
        """
        import re

        normalized = model.lower()
        # Strip trailing context-window suffixes (e.g. "-1m")
        normalized = re.sub(r"-\d+m$", "", normalized)
        # Replace dot between major.minor version with hyphen
        # e.g. "claude-opus-4.6" -> "claude-opus-4-6"
        normalized = re.sub(r"(\d+)\.(\d+)$", r"\1-\2", normalized)
        return normalized

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GitHub Copilot.
        Branches by model family to add family-specific params.
        """
        from litellm.utils import supports_reasoning

        base_params = super().get_supported_openai_params(model)

        if "claude" in model.lower():
            claude_model = self._normalize_claude_model_name(model)
            if supports_reasoning(model=claude_model):
                if "thinking" not in base_params:
                    base_params.append("thinking")
                if "reasoning_effort" not in base_params:
                    base_params.append("reasoning_effort")
            if "context_management" not in base_params:
                base_params.append("context_management")

        elif model.lower().startswith(("gpt-", "o1", "o3", "o4")):
            if "context_management" not in base_params:
                base_params.append("context_management")

        return base_params

    def _determine_initiator(self, messages: List[AllMessageValues]) -> str:
        """
        Determine if request is user or agent initiated based on message roles.
        Returns 'agent' if any message has role 'tool' or 'assistant', otherwise 'user'.
        """
        for message in messages:
            role = message.get("role")
            if role in ["tool", "assistant"]:
                return "agent"
        return "user"

    def _has_vision_content(self, messages: List[AllMessageValues]) -> bool:
        """
        Check if any message contains vision content (images).
        Returns True if any message has content with vision-related types, otherwise False.

        Checks for:
        - image_url content type (OpenAI format)
        - Content items with type 'image_url'
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Check if any content item indicates vision content
                for content_item in content:
                    if isinstance(content_item, dict):
                        # Check for image_url field (direct image URL)
                        if "image_url" in content_item:
                            return True
                        # Check for type field indicating image content
                        content_type = content_item.get("type")
                        if content_type == "image_url":
                            return True
        return False
