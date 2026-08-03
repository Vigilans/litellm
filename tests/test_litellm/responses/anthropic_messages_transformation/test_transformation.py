import json
import os

import pytest

import litellm
from litellm.responses.anthropic_messages_transformation.transformation import (
    translate_responses_request_to_anthropic_messages_request,
)

MODEL = "claude-opus-5"


def translate(input, **params):
    return translate_responses_request_to_anthropic_messages_request(
        model=MODEL, input=input, responses_api_request=params
    )


def user_message(text):
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def developer_message(text):
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text}],
    }


def assistant_message(text):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


class TestSystemAndMessages:
    def test_string_input_becomes_a_user_message(self):
        request = translate("hello").request
        assert request["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
        assert "system" not in request

    def test_instructions_lead_the_system_blocks(self):
        request = translate(
            [developer_message("dev rules")], instructions="top instructions"
        ).request
        assert request["system"] == [
            {"type": "text", "text": "top instructions"},
            {"type": "text", "text": "dev rules"},
        ]

    def test_developer_and_system_roles_both_lift_to_system(self):
        request = translate(
            [
                developer_message("a"),
                {"type": "message", "role": "system", "content": "b"},
                user_message("hi"),
            ]
        ).request
        assert request["system"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]
        assert [m["role"] for m in request["messages"]] == ["user"]

    def test_multi_part_developer_message_keeps_every_block(self):
        request = translate(
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "one"},
                        {"type": "input_text", "text": "two"},
                    ],
                }
            ]
        ).request
        assert [b["text"] for b in request["system"]] == ["one", "two"]

    def test_consecutive_same_role_items_are_merged(self):
        """Anthropic requires alternating roles; two user turns in a row is a 400."""
        request = translate(
            [user_message("a"), user_message("b"), assistant_message("c")]
        ).request
        assert request["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "c"}]},
        ]

    def test_mid_conversation_developer_message_does_not_split_user_turns(self):
        """Codex injects developer messages between user turns; lifting them must
        not leave two adjacent user messages behind."""
        request = translate(
            [user_message("a"), developer_message("switch"), user_message("b")]
        ).request
        assert len(request["messages"]) == 1
        assert [b["text"] for b in request["messages"][0]["content"]] == ["a", "b"]

    def test_empty_text_blocks_are_dropped(self):
        request = translate([developer_message(""), user_message("hi")]).request
        assert "system" not in request
        assert request["messages"][0]["content"] == [{"type": "text", "text": "hi"}]

    def test_message_with_only_dropped_content_produces_no_turn(self):
        request = translate(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio"}],
                },
                assistant_message("hi"),
            ]
        ).request
        assert [m["role"] for m in request["messages"]] == ["assistant"]

    def test_string_content_message(self):
        request = translate([{"role": "user", "content": "plain"}]).request
        assert request["messages"][0]["content"] == [{"type": "text", "text": "plain"}]

    def test_unknown_item_types_are_dropped(self):
        request = translate(
            [{"type": "image_generation_call", "id": "ig_1"}, user_message("hi")]
        ).request
        assert len(request["messages"]) == 1


class TestContentParts:
    def test_input_image_url_maps_to_image_block(self):
        request = translate(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": "https://example.com/a.png",
                        }
                    ],
                }
            ]
        ).request
        assert request["messages"][0]["content"][0] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/a.png"},
        }

    def test_input_image_file_id_maps_to_file_source(self):
        request = translate(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "detail": "auto", "file_id": "file-1"}
                    ],
                }
            ]
        ).request
        assert request["messages"][0]["content"][0] == {
            "type": "image",
            "source": {"type": "file", "file_id": "file-1"},
        }

    def test_input_file_maps_to_document_block(self):
        request = translate(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "file-2"}],
                }
            ]
        ).request
        assert request["messages"][0]["content"][0] == {
            "type": "document",
            "source": {"type": "file", "file_id": "file-2"},
        }

    def test_input_file_without_reference_is_dropped(self):
        request = translate(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "a.pdf"},
                        {"type": "input_text", "text": "read it"},
                    ],
                }
            ]
        ).request
        assert request["messages"][0]["content"] == [
            {"type": "text", "text": "read it"}
        ]


class TestOptionalParams:
    def test_max_output_tokens_maps_to_max_tokens(self):
        assert (
            translate([user_message("hi")], max_output_tokens=123).request["max_tokens"]
            == 123
        )

    def test_missing_max_output_tokens_falls_back_to_model_default(self):
        """max_tokens is required by Anthropic; Codex never sends it."""
        assert translate([user_message("hi")]).request["max_tokens"] > 0

    def test_reasoning_effort_is_passed_through(self):
        """AnthropicMessagesConfig maps this without a supports_reasoning gate,
        which is what unblocks unmapped Claude-compatible deployments."""
        assert (
            translate([user_message("hi")], reasoning={"effort": "max"}).request[
                "reasoning_effort"
            ]
            == "max"
        )

    def test_reasoning_without_effort_is_ignored(self):
        assert (
            "reasoning_effort"
            not in translate(
                [user_message("hi")], reasoning={"summary": "auto"}
            ).request
        )

    def test_sampling_and_stream_params_map_by_name(self):
        request = translate(
            [user_message("hi")], temperature=0.3, top_p=0.9, stream=True
        ).request
        assert (request["temperature"], request["top_p"], request["stream"]) == (
            0.3,
            0.9,
            True,
        )

    def test_json_schema_text_format_maps_to_output_config(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        request = translate(
            [user_message("hi")],
            text={"format": {"type": "json_schema", "name": "r", "schema": schema}},
        ).request
        assert request["output_config"] == {
            "format": {"type": "json_schema", "schema": schema}
        }

    def test_plain_text_format_is_ignored(self):
        assert (
            "output_config"
            not in translate(
                [user_message("hi")], text={"format": {"type": "text"}}
            ).request
        )

    def test_unsupported_params_are_not_forwarded(self):
        request = translate(
            [user_message("hi")],
            store=True,
            truncation="auto",
            safety_identifier="u1",
            include=["reasoning.encrypted_content"],
        ).request
        for key in ("store", "truncation", "safety_identifier", "include"):
            assert key not in request


class TestToolWiring:
    FUNCTION_TOOL = {
        "type": "function",
        "name": "wait",
        "parameters": {"type": "object"},
    }

    def test_top_level_tools_and_choice_are_translated(self):
        request = translate(
            [user_message("hi")], tools=[self.FUNCTION_TOOL], tool_choice="auto"
        ).request
        assert request["tools"][0]["name"] == "wait"
        assert request["tool_choice"] == {"type": "auto"}

    def test_additional_tools_item_is_hoisted_and_removed(self):
        result = translate(
            [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [self.FUNCTION_TOOL],
                },
                user_message("hi"),
            ],
            tool_choice="auto",
        )
        assert result.request["tools"][0]["name"] == "wait"
        assert len(result.request["messages"]) == 1

    def test_tool_choice_dropped_when_all_tools_are_dropped(self):
        """The live 400: `tools are required when tool choice is specified`."""
        request = translate(
            [user_message("hi")],
            tools=[{"type": "code_interpreter"}],
            tool_choice="auto",
        ).request
        assert "tools" not in request
        assert "tool_choice" not in request

    def test_parallel_tool_calls_becomes_disable_parallel_tool_use(self):
        request = translate(
            [user_message("hi")],
            tools=[self.FUNCTION_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
        ).request
        assert request["tool_choice"]["disable_parallel_tool_use"] is True

    def test_parallel_tool_calls_not_applied_to_none_choice(self):
        request = translate(
            [user_message("hi")],
            tools=[self.FUNCTION_TOOL],
            tool_choice="none",
            parallel_tool_calls=False,
        ).request
        assert request["tool_choice"] == {"type": "none"}

    def test_mcp_tool_moves_to_mcp_servers(self):
        request = translate(
            [user_message("hi")],
            tools=[
                {
                    "type": "mcp",
                    "server_label": "wiki",
                    "server_url": "https://mcp.example/sse",
                }
            ],
        ).request
        assert "tools" not in request
        assert request["mcp_servers"][0]["url"] == "https://mcp.example/sse"


class TestMultiTurnReplay:
    NAMESPACE_TOOL = {
        "type": "namespace",
        "name": "collab",
        "description": "d",
        "tools": [{"type": "function", "name": "spawn", "parameters": {}}],
    }
    CUSTOM_TOOL = {"type": "custom", "name": "exec"}

    def test_function_call_becomes_tool_use(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "wait",
                    "arguments": '{"seconds": 3}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            ],
            tools=[{"type": "function", "name": "wait", "parameters": {}}],
        ).request
        assert request["messages"][1] == {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "wait",
                    "input": {"seconds": 3},
                }
            ],
        }
        assert request["messages"][2]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "done",
        }

    def test_replayed_call_uses_the_flattened_tool_name(self):
        """A tool_use naming a tool absent from `tools` is a 400."""
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "spawn",
                    "namespace": "collab",
                    "arguments": "{}",
                },
            ],
            tools=[self.NAMESPACE_TOOL],
        ).request
        assert request["messages"][1]["content"][0]["name"] == "collab__spawn"

    def test_replayed_call_resolves_without_namespace_field(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "spawn",
                    "arguments": "{}",
                },
            ],
            tools=[self.NAMESPACE_TOOL],
        ).request
        assert request["messages"][1]["content"][0]["name"] == "collab__spawn"

    def test_custom_tool_call_roundtrips_through_the_input_field(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "custom_tool_call",
                    "call_id": "c2",
                    "name": "exec",
                    "input": "echo hi",
                },
                {"type": "custom_tool_call_output", "call_id": "c2", "output": "hi"},
            ],
            tools=[self.CUSTOM_TOOL],
        ).request
        assert request["messages"][1]["content"][0]["input"] == {"input": "echo hi"}
        assert request["messages"][2]["content"][0]["tool_use_id"] == "c2"

    def test_malformed_arguments_raise_instead_of_silently_emptying(self):
        with pytest.raises(litellm.exceptions.BadRequestError, match="not valid JSON"):
            translate(
                [
                    user_message("hi"),
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "wait",
                        "arguments": "{seconds:",
                    },
                ],
                tools=[{"type": "function", "name": "wait", "parameters": {}}],
            )

    def test_empty_arguments_string_is_an_empty_object(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "wait",
                    "arguments": "",
                },
            ],
            tools=[{"type": "function", "name": "wait", "parameters": {}}],
        ).request
        assert request["messages"][1]["content"][0]["input"] == {}

    def test_non_string_tool_output_is_json_encoded(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": {"ok": True},
                },
            ]
        ).request
        assert json.loads(request["messages"][0]["content"][-1]["content"]) == {
            "ok": True
        }

    def test_call_and_output_pair_do_not_collapse_into_one_turn(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "wait",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "wait",
                    "arguments": "{}",
                },
            ],
            tools=[{"type": "function", "name": "wait", "parameters": {}}],
        ).request
        assert [m["role"] for m in request["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]


class TestReasoningReplay:
    def test_reasoning_with_encrypted_content_becomes_a_thinking_block(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "thought"}],
                    "encrypted_content": "sig-abc",
                },
            ]
        ).request
        assert request["messages"][1]["content"][0] == {
            "type": "thinking",
            "thinking": "thought",
            "signature": "sig-abc",
        }

    def test_reasoning_content_preferred_over_summary(self):
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "summary"}],
                    "content": [{"type": "reasoning_text", "text": "full"}],
                    "encrypted_content": "sig",
                },
            ]
        ).request
        assert request["messages"][1]["content"][0]["thinking"] == "full"

    def test_reasoning_without_encrypted_content_is_dropped(self):
        """Anthropic rejects a thinking block with no signature, and Codex sends
        reasoning items with encrypted_content=null when it never got one."""
        request = translate(
            [
                user_message("hi"),
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": None,
                },
                assistant_message("hello"),
            ]
        ).request
        assert request["messages"][1]["content"] == [{"type": "text", "text": "hello"}]


class TestRealCodexCapture:
    """Driven by a captured Codex CLI request (prompt/description text trimmed):
    lite wire mode, no top-level tools, tool_choice=auto, developer messages
    interleaved between user turns, and a reasoning item with no signature."""

    @pytest.fixture(scope="class")
    def captured_request(self):
        with open(
            os.path.join(os.path.dirname(__file__), "codex_responses_request.json")
        ) as f:
            body = json.load(f)
        params = {k: v for k, v in body.items() if k not in ("input", "model")}
        return translate_responses_request_to_anthropic_messages_request(
            model=body["model"], input=body["input"], responses_api_request=params
        ).request

    def test_tools_and_tool_choice_survive(self, captured_request):
        assert len(captured_request["tools"]) == 9
        assert captured_request["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    def test_additional_tools_item_never_reaches_messages(self, captured_request):
        for message in captured_request["messages"]:
            for block in message["content"]:
                assert "additional_tools" not in json.dumps(block)

    def test_roles_alternate(self, captured_request):
        roles = [m["role"] for m in captured_request["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:]))

    def test_conversation_starts_with_user(self, captured_request):
        assert captured_request["messages"][0]["role"] == "user"

    def test_reasoning_effort_and_max_tokens_present(self, captured_request):
        assert captured_request["reasoning_effort"] == "max"
        assert captured_request["max_tokens"] > 0
