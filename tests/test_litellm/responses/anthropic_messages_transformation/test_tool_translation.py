import json

import pytest

from litellm.responses.anthropic_messages_transformation.tool_translation import (
    _ANTHROPIC_TOOL_NAME_MAX_LEN,
    _CUSTOM_TOOL_INPUT_FIELD,
    ToolTranslationContext,
    extract_additional_tools,
    translate_tool_choice,
    translate_tools,
    unwrap_custom_tool_input,
    wrap_custom_tool_input,
)

USER_MESSAGE = {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "hi"}],
}
FUNCTION_TOOL = {
    "type": "function",
    "name": "wait",
    "description": "Wait a while",
    "parameters": {
        "type": "object",
        "properties": {"seconds": {"type": "number"}},
        "required": ["seconds"],
    },
}
CUSTOM_TOOL = {
    "type": "custom",
    "name": "exec",
    "description": "Run a shell command",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: SOURCE\nSOURCE: /[\\s\\S]+/",
    },
}
NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "collaboration",
    "description": "Work with other agents",
    "tools": [
        {"type": "function", "name": "spawn_agent", "parameters": {"type": "object"}},
        {"type": "custom", "name": "message_agent"},
    ],
}


def anthropic_names(result):
    return [tool["name"] for tool in result.tools]


class TestExtractAdditionalTools:
    def test_tools_hoisted_and_item_removed(self):
        remaining, hoisted = extract_additional_tools(
            [
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [FUNCTION_TOOL, CUSTOM_TOOL],
                },
                USER_MESSAGE,
            ]
        )
        assert remaining == [USER_MESSAGE]
        assert hoisted == [FUNCTION_TOOL, CUSTOM_TOOL]

    def test_empty_additional_tools_item_is_still_removed(self):
        """An item left in `input` would be replayed to Claude as a conversation
        turn, so removal must key off seeing the item, not off hoisting a tool."""
        remaining, hoisted = extract_additional_tools(
            [
                {"type": "additional_tools", "role": "developer", "tools": []},
                USER_MESSAGE,
            ]
        )
        assert remaining == [USER_MESSAGE]
        assert hoisted == []

    def test_item_without_tools_key_is_removed(self):
        remaining, hoisted = extract_additional_tools(
            [{"type": "additional_tools", "role": "developer"}, USER_MESSAGE]
        )
        assert remaining == [USER_MESSAGE]
        assert hoisted == []

    def test_multiple_items_merge_in_order(self):
        first = {"type": "function", "name": "first", "parameters": {}}
        second = {"type": "function", "name": "second", "parameters": {}}
        remaining, hoisted = extract_additional_tools(
            [
                {"type": "additional_tools", "role": "developer", "tools": [first]},
                USER_MESSAGE,
                {"type": "additional_tools", "role": "developer", "tools": [second]},
            ]
        )
        assert remaining == [USER_MESSAGE]
        assert hoisted == [first, second]

    def test_string_input_passes_through(self):
        remaining, hoisted = extract_additional_tools("hello")
        assert remaining == "hello"
        assert hoisted == []

    def test_input_without_additional_tools_returns_same_list_object(self):
        items = [USER_MESSAGE, {"type": "function_call", "name": "wait"}]
        remaining, hoisted = extract_additional_tools(items)
        assert remaining is items
        assert hoisted == []


class TestTranslateTools:
    def test_function_tool_keeps_schema(self):
        result = translate_tools([FUNCTION_TOOL], None)
        assert result.tools == [
            {
                "name": "wait",
                "description": "Wait a while",
                "input_schema": {
                    "type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"],
                },
            }
        ]

    def test_function_tool_without_parameters_gets_object_schema(self):
        result = translate_tools([{"type": "function", "name": "ping"}], None)
        assert result.tools[0]["input_schema"] == {"type": "object"}

    def test_function_parameters_are_not_mutated(self):
        tool = {"type": "function", "name": "ping", "parameters": {}}
        translate_tools([tool], None)
        assert tool["parameters"] == {}

    def test_custom_tool_lowered_to_single_string_input(self):
        result = translate_tools([CUSTOM_TOOL], None)
        assert result.tools[0]["input_schema"] == {
            "type": "object",
            "properties": {
                _CUSTOM_TOOL_INPUT_FIELD: {
                    "type": "string",
                    "description": "The complete tool input, passed through verbatim.",
                }
            },
            "required": [_CUSTOM_TOOL_INPUT_FIELD],
            "additionalProperties": False,
        }

    def test_custom_grammar_is_serialized_into_description(self):
        description = translate_tools([CUSTOM_TOOL], None).tools[0]["description"]
        assert "Run a shell command" in description
        assert "lark grammar" in description
        assert "SOURCE: /[\\s\\S]+/" in description

    def test_custom_without_grammar_still_explains_the_string_field(self):
        description = translate_tools(
            [{"type": "custom", "name": "freeform"}], None
        ).tools[0]["description"]
        assert _CUSTOM_TOOL_INPUT_FIELD in description
        assert "grammar" not in description

    def test_namespace_flattened_with_separator(self):
        result = translate_tools([NAMESPACE_TOOL], None)
        assert anthropic_names(result) == [
            "collaboration__spawn_agent",
            "collaboration__message_agent",
        ]

    def test_namespace_description_prefixes_children(self):
        description = translate_tools([NAMESPACE_TOOL], None).tools[0]["description"]
        assert description.startswith(
            "[namespace: collaboration] Work with other agents"
        )

    def test_namespace_child_identity_records_namespace(self):
        context = translate_tools([NAMESPACE_TOOL], None).context
        identity = context.lookup_by_anthropic_name("collaboration__message_agent")
        assert identity.namespace == "collaboration"
        assert identity.response_name == "message_agent"
        assert identity.response_type == "custom"

    def test_nested_namespace_children_are_dropped(self):
        result = translate_tools(
            [
                {
                    "type": "namespace",
                    "name": "outer",
                    "description": "d",
                    "tools": [{"type": "namespace", "name": "inner", "tools": []}],
                }
            ],
            None,
        )
        assert result.tools == []

    def test_top_level_and_additional_tools_are_both_kept(self):
        result = translate_tools([FUNCTION_TOOL], [CUSTOM_TOOL])
        assert anthropic_names(result) == ["wait", "exec"]

    def test_duplicate_identity_kept_once(self):
        result = translate_tools([FUNCTION_TOOL], [FUNCTION_TOOL])
        assert anthropic_names(result) == ["wait"]

    def test_same_name_different_type_are_distinct_tools(self):
        result = translate_tools(
            [{"type": "function", "name": "run"}, {"type": "custom", "name": "run"}],
            None,
        )
        assert len(result.tools) == 2
        assert len(set(anthropic_names(result))) == 2

    def test_unsupported_tool_types_are_dropped(self):
        result = translate_tools(
            [
                {"type": "code_interpreter"},
                {"type": "image_generation"},
                {"type": "local_shell"},
                FUNCTION_TOOL,
            ],
            None,
        )
        assert anthropic_names(result) == ["wait"]

    def test_tool_without_name_is_dropped(self):
        result = translate_tools([{"type": "function", "parameters": {}}], None)
        assert result.tools == []

    def test_non_dict_entries_are_ignored(self):
        result = translate_tools(["function", None, FUNCTION_TOOL], None)
        assert anthropic_names(result) == ["wait"]

    def test_no_tools_yields_empty_result(self):
        result = translate_tools(None, None)
        assert result.tools == []
        assert result.mcp_servers == []


class TestWebSearchAndMcp:
    def test_web_search_maps_to_native_tool(self):
        result = translate_tools([{"type": "web_search"}], None)
        assert result.tools == [{"type": "web_search_20250305", "name": "web_search"}]

    def test_versioned_web_search_type_is_recognized(self):
        result = translate_tools([{"type": "web_search_2025_08_26"}], None)
        assert result.tools[0]["name"] == "web_search"

    def test_web_search_preview_is_recognized(self):
        result = translate_tools([{"type": "web_search_preview"}], None)
        assert result.tools[0]["name"] == "web_search"

    def test_flat_user_location_is_renested_for_anthropic(self):
        """Responses declares user_location flat; AnthropicConfig reads
        user_location.approximate, so a straight pass-through loses the city."""
        result = translate_tools(
            [
                {
                    "type": "web_search",
                    "user_location": {
                        "type": "approximate",
                        "city": "Seattle",
                        "country": "US",
                    },
                }
            ],
            None,
        )
        assert result.tools[0]["user_location"] == {
            "type": "approximate",
            "city": "Seattle",
            "country": "US",
        }

    def test_search_context_size_maps_to_max_uses(self):
        result = translate_tools(
            [{"type": "web_search", "search_context_size": "high"}], None
        )
        assert result.tools[0]["max_uses"] > 0

    def test_mcp_tool_moves_to_mcp_servers(self):
        result = translate_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "deepwiki",
                    "server_url": "https://mcp.example/sse",
                    "allowed_tools": ["ask"],
                    "headers": {"Authorization": "Bearer sk-123"},
                }
            ],
            None,
        )
        assert result.tools == []
        assert result.mcp_servers == [
            {
                "type": "url",
                "url": "https://mcp.example/sse",
                "name": "deepwiki",
                "tool_configuration": {"allowed_tools": ["ask"]},
                "authorization_token": "sk-123",
            }
        ]

    def test_connector_only_mcp_tool_is_dropped(self):
        result = translate_tools(
            [
                {
                    "type": "mcp",
                    "server_label": "gmail",
                    "connector_id": "connector_gmail",
                }
            ],
            None,
        )
        assert result.mcp_servers == []


class TestNameSanitization:
    def test_illegal_characters_replaced(self):
        result = translate_tools([{"type": "function", "name": "mcp.ologs/get"}], None)
        assert anthropic_names(result) == ["mcp_ologs_get"]

    def test_sanitized_collision_produces_distinct_names(self):
        """`a.b` and `a/b` both sanitize to `a_b`; the second must not silently
        overwrite the first's context entry."""
        result = translate_tools(
            [{"type": "function", "name": "a.b"}, {"type": "function", "name": "a/b"}],
            None,
        )
        names = anthropic_names(result)
        assert len(set(names)) == 2
        assert result.context.lookup_by_anthropic_name(names[0]).response_name == "a.b"
        assert result.context.lookup_by_anthropic_name(names[1]).response_name == "a/b"

    def test_flattened_name_colliding_with_top_level_tool_is_disambiguated(self):
        result = translate_tools(
            [
                {"type": "function", "name": "ns__child"},
                {
                    "type": "namespace",
                    "name": "ns",
                    "description": "d",
                    "tools": [{"type": "function", "name": "child"}],
                },
            ],
            None,
        )
        assert len(set(anthropic_names(result))) == 2

    def test_long_name_is_truncated_within_limit(self):
        result = translate_tools([{"type": "function", "name": "x" * 200}], None)
        assert len(result.tools[0]["name"]) == _ANTHROPIC_TOOL_NAME_MAX_LEN

    def test_long_names_sharing_a_prefix_stay_distinct(self):
        prefix = "y" * 80
        result = translate_tools(
            [
                {"type": "function", "name": prefix + "alpha"},
                {"type": "function", "name": prefix + "beta"},
            ],
            None,
        )
        assert len(set(anthropic_names(result))) == 2

    def test_name_is_deterministic_across_calls(self):
        """The context is rebuilt every turn, so a replayed tool call only matches
        if the same tool list always produces the same names."""
        tools = [{"type": "function", "name": "z" * 100}, NAMESPACE_TOOL]
        assert anthropic_names(translate_tools(tools, None)) == anthropic_names(
            translate_tools(tools, None)
        )

    def test_name_does_not_depend_on_position(self):
        a = {"type": "function", "name": "w" * 100}
        b = {"type": "function", "name": "ordinary"}
        first = translate_tools([a, b], None)
        second = translate_tools([b, a], None)
        assert (
            first.context.resolve_response_tool("w" * 100, "function").anthropic_name
            == second.context.resolve_response_tool(
                "w" * 100, "function"
            ).anthropic_name
        )

    def test_name_of_only_illegal_characters_falls_back(self):
        result = translate_tools([{"type": "function", "name": "..."}], None)
        assert result.tools[0]["name"] == "___"


class TestTranslateToolChoice:
    def context_with(self, *tools):
        return translate_tools(list(tools), None).context

    def test_auto_and_required_and_none(self):
        context = self.context_with(FUNCTION_TOOL)
        assert translate_tool_choice("auto", context, True) == {"type": "auto"}
        assert translate_tool_choice("required", context, True) == {"type": "any"}
        assert translate_tool_choice("none", context, True) == {"type": "none"}

    def test_named_function_choice_uses_mapped_name(self):
        context = self.context_with(NAMESPACE_TOOL)
        assert translate_tool_choice(
            {"type": "function", "name": "spawn_agent"}, context, True
        ) == {"type": "tool", "name": "collaboration__spawn_agent"}

    def test_ambiguous_bare_name_across_namespaces_is_dropped(self):
        """`tool_choice` has no namespace field, so a bare name matching two
        namespaces cannot be resolved; guessing would call the wrong tool."""
        context = self.context_with(
            {
                "type": "namespace",
                "name": "alpha",
                "description": "d",
                "tools": [{"type": "function", "name": "run"}],
            },
            {
                "type": "namespace",
                "name": "beta",
                "description": "d",
                "tools": [{"type": "function", "name": "run"}],
            },
        )
        assert (
            translate_tool_choice({"type": "function", "name": "run"}, context, True)
            is None
        )

    def test_top_level_tool_wins_over_namespaced_same_name(self):
        context = self.context_with(
            {"type": "function", "name": "run"},
            {
                "type": "namespace",
                "name": "alpha",
                "description": "d",
                "tools": [{"type": "function", "name": "run"}],
            },
        )
        assert translate_tool_choice(
            {"type": "function", "name": "run"}, context, True
        ) == {"type": "tool", "name": "run"}

    def test_named_custom_choice_uses_mapped_name(self):
        context = self.context_with({"type": "custom", "name": "mcp.exec"})
        assert translate_tool_choice(
            {"type": "custom", "name": "mcp.exec"}, context, True
        ) == {"type": "tool", "name": "mcp_exec"}

    def test_choice_naming_a_dropped_tool_is_removed(self):
        """Sending tool_choice for a tool that isn't in `tools` is a 400."""
        context = self.context_with(FUNCTION_TOOL)
        assert (
            translate_tool_choice({"type": "function", "name": "gone"}, context, True)
            is None
        )

    def test_choice_type_must_match_the_tool_type(self):
        context = self.context_with(FUNCTION_TOOL)
        assert (
            translate_tool_choice({"type": "custom", "name": "wait"}, context, True)
            is None
        )

    def test_choice_dropped_when_no_tool_survived(self):
        """This is the live 400: Codex sends tool_choice=auto, every tool is
        dropped, and the bare tool_choice reaches Anthropic."""
        context = ToolTranslationContext()
        assert translate_tool_choice("auto", context, False) is None
        assert (
            translate_tool_choice({"type": "function", "name": "wait"}, context, False)
            is None
        )

    def test_allowed_tools_collapses_to_auto_or_any(self):
        context = self.context_with(FUNCTION_TOOL)
        assert translate_tool_choice(
            {"type": "allowed_tools", "mode": "auto", "tools": []}, context, True
        ) == {"type": "auto"}
        assert translate_tool_choice(
            {"type": "allowed_tools", "mode": "required", "tools": []}, context, True
        ) == {"type": "any"}

    @pytest.mark.parametrize(
        "choice", [None, "nonsense", {"type": "mcp", "server_label": "x"}, 7]
    )
    def test_unsupported_choices_are_dropped(self, choice):
        assert (
            translate_tool_choice(choice, self.context_with(FUNCTION_TOOL), True)
            is None
        )


class TestCustomToolInputRoundtrip:
    def test_wrap_then_unwrap_is_lossless(self):
        raw = '// @exec: sh\necho "hi" $VAR\n'
        assert unwrap_custom_tool_input(wrap_custom_tool_input(raw)) == raw

    def test_extra_keys_fall_back_to_full_json(self):
        """The model ignoring the schema must not cost us its output."""
        payload = {"input": "ls", "cwd": "/tmp"}
        assert json.loads(unwrap_custom_tool_input(payload)) == payload

    def test_non_string_input_field_falls_back_to_full_json(self):
        payload = {"input": {"cmd": "ls"}}
        assert json.loads(unwrap_custom_tool_input(payload)) == payload

    def test_plain_string_passes_through(self):
        assert unwrap_custom_tool_input("ls -al") == "ls -al"

    def test_unicode_is_not_escaped(self):
        assert unwrap_custom_tool_input({"a": "中文"}) == '{"a":"中文"}'
