import json

from litellm.responses.anthropic_messages_transformation.tool_translation import (
    translate_tools,
)
from litellm.responses.anthropic_messages_transformation.transformation import (
    translate_anthropic_messages_response_to_responses_api_response,
)

FUNCTION_TOOL = {"type": "function", "name": "wait", "parameters": {"type": "object"}}
CUSTOM_TOOL = {"type": "custom", "name": "exec"}
NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "collab",
    "description": "d",
    "tools": [{"type": "function", "name": "spawn", "parameters": {}}],
}


def anthropic_response(content, **overrides):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        **overrides,
    }


def translate(content, tools=None, request=None, **overrides):
    context = translate_tools(tools or [], None).context
    return translate_anthropic_messages_response_to_responses_api_response(
        response=anthropic_response(content, **overrides),
        tool_context=context,
        request=request or {},
        created_at=1700000000,
    )


def output_items(content, **kwargs):
    """``ResponsesAPIResponse`` coerces the raw output dicts into the openai SDK
    item models, so the discriminated union picking the right model is itself
    part of what these assertions check."""
    return [item.model_dump() for item in translate(content, **kwargs).output]


class TestContentBlockMapping:
    def test_text_becomes_a_message_item(self):
        output = output_items([{"type": "text", "text": "hello"}])
        assert output[0]["type"] == "message"
        assert output[0]["id"] == "msg_1_0"
        assert output[0]["role"] == "assistant"
        assert output[0]["status"] == "completed"
        assert output[0]["content"][0]["type"] == "output_text"
        assert output[0]["content"][0]["text"] == "hello"

    def test_thinking_signature_lands_in_encrypted_content(self):
        """The client replays this field, and Anthropic rejects a thinking block
        whose signature is missing on the next turn."""
        output = output_items(
            [{"type": "thinking", "thinking": "hmm", "signature": "sig-abc"}]
        )
        assert output[0]["type"] == "reasoning"
        assert output[0]["encrypted_content"] == "sig-abc"
        assert output[0]["summary"] == [{"type": "summary_text", "text": "hmm"}]

    def test_redacted_thinking_carries_its_data(self):
        output = output_items([{"type": "redacted_thinking", "data": "enc"}])
        assert output[0]["type"] == "reasoning"
        assert output[0]["encrypted_content"] == "enc"
        assert output[0]["summary"] == []

    def test_block_order_is_preserved(self):
        """Interleaved thinking/text/tool_use is the shape Claude actually emits;
        reordering changes what the client replays next turn."""
        output = output_items(
            [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "text", "text": "a"},
                {"type": "tool_use", "id": "tu_1", "name": "wait", "input": {}},
                {"type": "text", "text": "b"},
            ],
            tools=[FUNCTION_TOOL],
        )
        assert [item["type"] for item in output] == [
            "reasoning",
            "message",
            "function_call",
            "message",
        ]

    def test_unknown_blocks_are_dropped_without_losing_the_rest(self):
        output = output_items(
            [
                {"type": "server_tool_use", "id": "st_1", "name": "web_search"},
                {"type": "text", "text": "kept"},
            ]
        )
        assert len(output) == 1
        assert output[0]["content"][0]["text"] == "kept"

    def test_item_ids_are_unique(self):
        output = output_items(
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        )
        assert output[0]["id"] != output[1]["id"]


class TestToolUseRestoration:
    def test_function_tool_use_becomes_function_call(self):
        output = output_items(
            [{"type": "tool_use", "id": "tu_1", "name": "wait", "input": {"s": 3}}],
            tools=[FUNCTION_TOOL],
        )
        assert output[0]["type"] == "function_call"
        assert output[0]["call_id"] == "tu_1"
        assert output[0]["name"] == "wait"
        assert json.loads(output[0]["arguments"]) == {"s": 3}

    def test_custom_tool_use_becomes_custom_tool_call_with_raw_input(self):
        output = output_items(
            [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "exec",
                    "input": {"input": "echo hi"},
                }
            ],
            tools=[CUSTOM_TOOL],
        )
        assert output[0]["type"] == "custom_tool_call"
        assert output[0]["input"] == "echo hi"
        assert output[0]["name"] == "exec"

    def test_custom_tool_input_off_schema_falls_back_to_json(self):
        output = output_items(
            [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "exec",
                    "input": {"cmd": "ls", "cwd": "/"},
                }
            ],
            tools=[CUSTOM_TOOL],
        )
        assert json.loads(output[0]["input"]) == {"cmd": "ls", "cwd": "/"}

    def test_namespaced_tool_use_restores_name_and_namespace(self):
        output = output_items(
            [{"type": "tool_use", "id": "tu_1", "name": "collab__spawn", "input": {}}],
            tools=[NAMESPACE_TOOL],
        )
        assert output[0]["name"] == "spawn"
        assert output[0]["namespace"] == "collab"

    def test_unknown_tool_name_falls_back_to_function_call(self):
        """A server-side tool we never mapped must still reach the client rather
        than disappear from the output."""
        output = output_items(
            [{"type": "tool_use", "id": "tu_1", "name": "surprise", "input": {"a": 1}}],
            tools=[FUNCTION_TOOL],
        )
        assert output[0]["type"] == "function_call"
        assert output[0]["name"] == "surprise"

    def test_missing_input_becomes_empty_arguments_object(self):
        output = output_items(
            [{"type": "tool_use", "id": "tu_1", "name": "wait"}], tools=[FUNCTION_TOOL]
        )
        assert output[0]["arguments"] == "{}"


class TestStatusAndUsage:
    def test_end_turn_is_completed(self):
        assert translate([], stop_reason="end_turn").status == "completed"

    def test_tool_use_stop_reason_is_completed(self):
        assert translate([], stop_reason="tool_use").status == "completed"

    def test_max_tokens_is_incomplete_with_a_reason(self):
        response = translate([], stop_reason="max_tokens")
        assert response.status == "incomplete"
        assert response.incomplete_details.reason == "max_output_tokens"

    def test_completed_response_has_no_incomplete_details(self):
        assert translate([], stop_reason="end_turn").incomplete_details is None

    def test_unknown_stop_reason_defaults_to_completed(self):
        assert translate([], stop_reason="something_new").status == "completed"

    def test_usage_totals_and_cache_details(self):
        usage = translate(
            [],
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 5,
            },
        ).usage
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (
            100,
            20,
            120,
        )
        assert usage.input_tokens_details.cached_tokens == 80
        assert usage.input_tokens_details.cache_creation_tokens == 5

    def test_cache_tokens_never_land_in_output_details(self):
        usage = translate(
            [],
            usage={"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 9},
        ).usage
        assert usage.output_tokens_details.reasoning_tokens == 0

    def test_missing_usage_is_none(self):
        assert translate([], usage=None).usage is None


class TestRequestEcho:
    def test_original_responses_tools_are_advertised(self):
        """The client must see the tools it sent, not the lowered Anthropic
        schemas (which erase grammars and flatten namespaces)."""
        grammar_tool = {
            "type": "custom",
            "name": "exec",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: X"},
        }
        response = translate(
            [],
            tools=[grammar_tool],
            request={"tools": [grammar_tool], "tool_choice": "auto"},
        )
        advertised = response.tools[0].model_dump(exclude_none=True)
        assert advertised["type"] == "custom"
        assert advertised["name"] == "exec"
        assert advertised["format"] == grammar_tool["format"]
        assert response.tool_choice == "auto"

    def test_request_params_are_echoed(self):
        response = translate(
            [],
            request={
                "instructions": "be brief",
                "temperature": 0.4,
                "top_p": 0.9,
                "max_output_tokens": 512,
                "reasoning": {"effort": "high"},
                "metadata": {"k": "v"},
                "user": "u1",
                "store": False,
            },
        )
        assert response.instructions == "be brief"
        assert (response.temperature, response.top_p) == (0.4, 0.9)
        assert response.max_output_tokens == 512
        assert response.reasoning == {"effort": "high"}
        assert response.metadata == {"k": "v"}
        assert response.user == "u1"
        assert response.store is False

    def test_identity_fields_come_from_the_anthropic_response(self):
        response = translate([{"type": "text", "text": "hi"}])
        assert response.id == "msg_1"
        assert response.model == "claude-opus-5"
        assert response.object == "response"
        assert response.created_at == 1700000000
