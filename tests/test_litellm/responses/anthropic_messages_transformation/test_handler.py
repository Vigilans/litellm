import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.anthropic_messages_transformation.handler import (
    LiteLLMResponsesToMessagesAPIHandler,
)
from litellm.responses.main import _should_route_responses_to_anthropic_messages
from litellm.responses.streaming_iterator import ResponsesAPIEventStreamIterator

HANDLER_PATH = (
    "litellm.llms.anthropic.experimental_pass_through.messages.handler"
    ".anthropic_messages_handler"
)

ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}


def user_input(text="hello"):
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
    ]


class TestRouting:
    def route(
        self,
        model,
        provider,
        native_config=None,
        use_chat_completions_api=False,
        is_async=True,
    ):
        return _should_route_responses_to_anthropic_messages(
            model=model,
            custom_llm_provider=provider,
            responses_api_provider_config=native_config,
            use_chat_completions_api=use_chat_completions_api,
            is_async=is_async,
        )

    def test_anthropic_claude_is_routed(self):
        assert self.route("claude-opus-5", "anthropic") is True

    def test_unmapped_model_under_anthropic_provider_is_routed(self):
        """`anthropic/moonshot/kimi-k3` has no model-map entry, which is exactly
        why reasoning_effort 400s on the chat bridge."""
        assert self.route("moonshot/kimi-k3", "anthropic") is True

    def test_copilot_claude_is_routed(self):
        assert self.route("claude-sonnet-4.5", "github_copilot") is True

    def test_vertex_claude_is_routed(self):
        assert self.route("claude-3-5-sonnet-v2@20241022", "vertex_ai") is True

    def test_copilot_gpt_is_not_routed(self):
        """Same provider, non-Claude model: no native /v1/messages config."""
        assert self.route("gpt-5.6", "github_copilot") is False

    def test_vertex_gemini_is_not_routed(self):
        assert self.route("gemini-2.5-pro", "vertex_ai") is False

    def test_openai_is_not_routed(self):
        assert self.route("gpt-4o", "openai") is False

    def test_native_responses_config_takes_priority(self):
        assert (
            self.route(
                "claude-opus-5", "anthropic", native_config=OpenAIResponsesAPIConfig()
            )
            is False
        )

    def test_per_request_chat_bridge_opt_in_wins(self):
        assert (
            self.route("claude-opus-5", "anthropic", use_chat_completions_api=True)
            is False
        )

    def test_unknown_provider_string_is_not_routed(self):
        """The provider string is used to construct an enum member; an arbitrary
        string must not raise."""
        assert self.route("claude-opus-5", "definitely-not-a-provider") is False

    def test_missing_provider_is_not_routed(self):
        assert self.route("claude-opus-5", None) is False

    def test_sync_calls_keep_the_chat_bridge(self):
        """`/v1/messages` has no sync transport, so sync `litellm.responses()`
        must keep working through the chat bridge rather than start erroring."""
        assert self.route("claude-opus-5", "anthropic", is_async=False) is False

    def test_global_escape_hatch_reverts_to_the_chat_bridge(self, monkeypatch):
        monkeypatch.setattr(
            litellm, "use_chat_completions_url_for_anthropic_responses", True
        )
        assert self.route("claude-opus-5", "anthropic") is False


class TestHandlerDispatch:
    @pytest.mark.asyncio
    async def test_translated_request_reaches_the_messages_handler(self):
        with patch(
            HANDLER_PATH, new=AsyncMock(return_value=ANTHROPIC_RESPONSE)
        ) as mock:
            await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                model="claude-opus-5",
                input=user_input(),
                responses_api_request={
                    "instructions": "be brief",
                    "max_output_tokens": 100,
                    "reasoning": {"effort": "high"},
                    "tools": [{"type": "function", "name": "wait", "parameters": {}}],
                    "tool_choice": "auto",
                },
                custom_llm_provider="anthropic",
            )
        sent = mock.call_args.kwargs
        assert sent["model"] == "claude-opus-5"
        assert sent["custom_llm_provider"] == "anthropic"
        assert sent["max_tokens"] == 100
        assert sent["reasoning_effort"] == "high"
        assert sent["system"] == [{"type": "text", "text": "be brief"}]
        assert sent["tools"][0]["name"] == "wait"
        assert sent["tool_choice"] == {"type": "auto"}
        assert sent["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_response_is_translated_back(self):
        with patch(HANDLER_PATH, new=AsyncMock(return_value=ANTHROPIC_RESPONSE)):
            response = (
                await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                    model="claude-opus-5",
                    input=user_input(),
                    responses_api_request={},
                    custom_llm_provider="anthropic",
                )
            )
        assert response.id == "msg_1"
        assert response.status == "completed"
        assert response.output[0].content[0].text == "hi"

    @pytest.mark.asyncio
    async def test_model_and_stream_are_not_duplicated(self):
        """The translated request carries model/stream too; passing both raises
        TypeError for a duplicate keyword."""
        with patch(
            HANDLER_PATH, new=AsyncMock(return_value=ANTHROPIC_RESPONSE)
        ) as mock:
            await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                model="claude-opus-5",
                input=user_input(),
                responses_api_request={"stream": False},
                custom_llm_provider="anthropic",
                stream=False,
            )
        assert mock.call_args.kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_extra_kwargs_are_forwarded(self):
        with patch(
            HANDLER_PATH, new=AsyncMock(return_value=ANTHROPIC_RESPONSE)
        ) as mock:
            await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                model="claude-opus-5",
                input=user_input(),
                responses_api_request={},
                custom_llm_provider="anthropic",
                api_key="sk-test",
                api_base="https://example.test",
                timeout=30.0,
                extra_headers={"x-trace": "1"},
            )
        sent = mock.call_args.kwargs
        assert sent["api_key"] == "sk-test"
        assert sent["api_base"] == "https://example.test"
        assert sent["timeout"] == 30.0
        assert sent["extra_headers"] == {"x-trace": "1"}

    @pytest.mark.asyncio
    async def test_async_flag_is_set_so_the_handler_returns_a_coroutine(self):
        with patch(
            HANDLER_PATH, new=AsyncMock(return_value=ANTHROPIC_RESPONSE)
        ) as mock:
            await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                model="claude-opus-5",
                input=user_input(),
                responses_api_request={},
                custom_llm_provider="anthropic",
            )
        assert mock.call_args.kwargs["is_async"] is True

    @pytest.mark.asyncio
    async def test_streaming_returns_the_translating_iterator(self):
        async def anthropic_sse():
            for chunk in [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "model": "claude-opus-5",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {},
                },
                {"type": "message_stop"},
            ]:
                yield f"data: {json.dumps(chunk)}\n\n".encode()

        with patch(HANDLER_PATH, new=AsyncMock(return_value=anthropic_sse())):
            stream = (
                await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                    model="claude-opus-5",
                    input=user_input(),
                    responses_api_request={},
                    custom_llm_provider="anthropic",
                    stream=True,
                )
            )
        events = [event async for event in stream]
        assert events[0].type.value == "response.created"
        assert events[-1].type.value == "response.completed"
        assert events[-1].response.output[0].content[0].text == "hi"

    @pytest.mark.asyncio
    async def test_actual_stream_result_uses_logging_lifecycle(self):
        async def anthropic_sse():
            yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': 'msg_1', 'model': 'claude-opus-5', 'usage': {'input_tokens': 1, 'output_tokens': 0}}})}\n\n".encode()
            yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {}})}\n\n".encode()
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n".encode()

        logger = MagicMock()
        with patch(HANDLER_PATH, new=AsyncMock(return_value=anthropic_sse())):
            stream = await LiteLLMResponsesToMessagesAPIHandler.async_response_api_handler(
                model="claude-opus-5",
                input=user_input(),
                responses_api_request={},
                custom_llm_provider="anthropic",
                stream=False,
                litellm_logging_obj=logger,
            )

        assert isinstance(stream, ResponsesAPIEventStreamIterator)

    def test_sync_calls_are_rejected(self):
        """The underlying /v1/messages handler has no sync path; failing loudly
        beats a confusing error from deep inside the HTTP layer."""
        with pytest.raises(ValueError, match="async only"):
            LiteLLMResponsesToMessagesAPIHandler.response_api_handler(
                model="claude-opus-5",
                input=user_input(),
                responses_api_request={},
                custom_llm_provider="anthropic",
                _is_async=False,
            )


class TestEndToEndThroughAresponses:
    """Drives the public entry point with the captured Codex request, which is
    the request that currently 400s with `tools are required when tool choice is
    specified`."""

    @pytest.fixture(scope="class")
    def capture(self):
        import os

        with open(
            os.path.join(os.path.dirname(__file__), "codex_responses_request.json")
        ) as f:
            return json.load(f)

    @pytest.fixture
    async def dispatched(self, capture):
        anthropic_response = {
            **ANTHROPIC_RESPONSE,
            "content": [
                {"type": "thinking", "thinking": "think", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "exec",
                    "input": {"input": "ls -al"},
                },
            ],
            "stop_reason": "tool_use",
        }
        params = {
            k: v for k, v in capture.items() if k not in ("input", "model", "stream")
        }
        with patch(
            HANDLER_PATH, new=AsyncMock(return_value=anthropic_response)
        ) as mock:
            response = await litellm.aresponses(
                model="anthropic/claude-opus-5",
                input=capture["input"],
                api_key="sk-fake",
                **params,
            )
        return mock.call_args.kwargs, response

    @pytest.mark.asyncio
    async def test_request_reaches_the_messages_endpoint_with_tools(self, dispatched):
        sent, _ = dispatched
        assert [t["name"] for t in sent["tools"]] == [
            "exec",
            "wait",
            "request_user_input",
            "collaboration__followup_task",
            "collaboration__interrupt_agent",
            "collaboration__list_agents",
            "collaboration__send_message",
            "collaboration__spawn_agent",
            "collaboration__wait_agent",
        ]
        assert sent["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    @pytest.mark.asyncio
    async def test_reasoning_effort_survives_for_the_captured_request(self, dispatched):
        sent, _ = dispatched
        assert sent["reasoning_effort"] == "max"

    @pytest.mark.asyncio
    async def test_developer_messages_became_system_and_roles_alternate(
        self, dispatched
    ):
        sent, _ = dispatched
        assert len(sent["system"]) == 11
        roles = [m["role"] for m in sent["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:]))

    @pytest.mark.asyncio
    async def test_response_restores_custom_tool_and_signature(self, dispatched):
        _, response = dispatched
        assert [item.type for item in response.output] == [
            "reasoning",
            "custom_tool_call",
        ]
        assert response.output[0].encrypted_content == "sig"
        assert response.output[1].input == "ls -al"

    @pytest.mark.asyncio
    async def test_response_advertises_the_tools_sent_via_additional_tools(
        self, dispatched
    ):
        """Codex sends no top-level `tools`, so echoing only that param reports
        zero tools back for a request that in fact had nine."""
        _, response = dispatched
        assert [t.name for t in response.tools] == [
            "exec",
            "wait",
            "request_user_input",
            "collaboration",
        ]
        assert response.tools[0].type == "custom"
        assert response.tools[3].type == "namespace"


class TestSyncFallsThroughToTheChatBridge:
    @patch(
        "litellm.responses.main.litellm_completion_transformation_handler.response_api_handler"
    )
    def test_sync_claude_request_still_reaches_the_chat_bridge(self, mock_chat_bridge):
        """Sync `litellm.responses()` against a Claude model worked before this
        bridge existed and must keep working; `/v1/messages` is async only."""
        mock_chat_bridge.return_value = MagicMock()

        litellm.responses(
            model="anthropic/claude-3-haiku",
            input="Hello",
            litellm_logging_obj=MagicMock(),
        )

        mock_chat_bridge.assert_called_once()
