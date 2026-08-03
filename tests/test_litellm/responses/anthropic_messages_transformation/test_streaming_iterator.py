import json

import pytest

from litellm.responses.anthropic_messages_transformation.streaming_iterator import (
    AnthropicMessagesToResponsesStreamIterator,
)
from litellm.responses.anthropic_messages_transformation.tool_translation import (
    translate_tools,
)

FUNCTION_TOOL = {"type": "function", "name": "wait", "parameters": {"type": "object"}}
CUSTOM_TOOL = {"type": "custom", "name": "exec"}
NAMESPACE_TOOL = {
    "type": "namespace",
    "name": "collab",
    "description": "d",
    "tools": [{"type": "function", "name": "spawn", "parameters": {}}],
}

MESSAGE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 12, "output_tokens": 0},
    },
}


def message_delta(stop_reason="end_turn", output_tokens=7):
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


def text_block(index, chunks):
    yield {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }
    for chunk in chunks:
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": chunk},
        }
    yield {"type": "content_block_stop", "index": index}


def thinking_block(index, chunks, signature_chunks=("sig-a", "sig-b")):
    yield {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    }
    for chunk in chunks:
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": chunk},
        }
    for chunk in signature_chunks:
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": chunk},
        }
    yield {"type": "content_block_stop", "index": index}


def tool_use_block(index, name, json_chunks, call_id="tu_1"):
    yield {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
    }
    for chunk in json_chunks:
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": chunk},
        }
    yield {"type": "content_block_stop", "index": index}


async def byte_stream(chunks):
    for chunk in chunks:
        yield f"event: {chunk['type']}\ndata: {json.dumps(chunk)}\n\n".encode()


async def collect(chunks, tools=None, request=None):
    context = translate_tools(tools or [], None).context
    iterator = AnthropicMessagesToResponsesStreamIterator(
        anthropic_stream=byte_stream(chunks),
        model="claude-opus-5",
        tool_context=context,
        responses_api_request=request or {},
        created_at=1700000000,
    )
    return [event async for event in iterator]


def types_of(events):
    return [event.type.value for event in events]


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_message_start_emits_created_then_in_progress(self):
        events = await collect([MESSAGE_START])
        assert types_of(events) == ["response.created", "response.in_progress"]
        assert events[0].response.id == "msg_1"
        assert events[0].response.status == "in_progress"

    @pytest.mark.asyncio
    async def test_message_stop_emits_completed_with_final_output(self):
        events = await collect(
            [
                MESSAGE_START,
                *text_block(0, ["hi"]),
                message_delta(),
                {"type": "message_stop"},
            ]
        )
        completed = events[-1]
        assert completed.type == "response.completed"
        assert completed.response.status == "completed"
        assert completed.response.output[0].content[0].text == "hi"

    @pytest.mark.asyncio
    async def test_max_tokens_ends_with_incomplete(self):
        events = await collect(
            [
                MESSAGE_START,
                *text_block(0, ["partial"]),
                message_delta(stop_reason="max_tokens"),
                {"type": "message_stop"},
            ]
        )
        assert events[-1].type == "response.incomplete"
        assert events[-1].response.incomplete_details.reason == "max_output_tokens"

    @pytest.mark.asyncio
    async def test_usage_is_accumulated_across_start_and_delta(self):
        events = await collect(
            [MESSAGE_START, message_delta(output_tokens=42), {"type": "message_stop"}]
        )
        usage = events[-1].response.usage
        assert usage.input_tokens == 12
        assert usage.output_tokens == 42

    @pytest.mark.asyncio
    async def test_sequence_numbers_are_monotonic(self):
        events = await collect(
            [
                MESSAGE_START,
                *thinking_block(0, ["t"]),
                *text_block(1, ["a"]),
                message_delta(),
                {"type": "message_stop"},
            ]
        )
        numbers = [event.sequence_number for event in events]
        assert numbers == sorted(numbers)
        assert len(set(numbers)) == len(numbers)

    @pytest.mark.asyncio
    async def test_error_chunk_becomes_an_error_event(self):
        events = await collect(
            [
                MESSAGE_START,
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
            ]
        )
        assert events[-1].type == "error"
        assert events[-1].error.message == "Overloaded"


class TestTextBlocks:
    @pytest.mark.asyncio
    async def test_text_block_event_sequence(self):
        events = await collect([MESSAGE_START, *text_block(0, ["he", "llo"])])
        assert types_of(events)[2:] == [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]

    @pytest.mark.asyncio
    async def test_deltas_are_forwarded_verbatim_and_accumulated(self):
        events = await collect([MESSAGE_START, *text_block(0, ["he", "llo"])])
        deltas = [e.delta for e in events if e.type == "response.output_text.delta"]
        assert deltas == ["he", "llo"]
        done = next(e for e in events if e.type == "response.output_text.done")
        assert done.text == "hello"

    @pytest.mark.asyncio
    async def test_two_text_blocks_get_distinct_output_indexes(self):
        events = await collect(
            [MESSAGE_START, *text_block(0, ["a"]), *text_block(1, ["b"])]
        )
        added = [e for e in events if e.type == "response.output_item.added"]
        assert [e.output_index for e in added] == [0, 1]
        assert added[0].item.id != added[1].item.id


class TestThinkingBlocks:
    @pytest.mark.asyncio
    async def test_thinking_block_event_sequence(self):
        events = await collect([MESSAGE_START, *thinking_block(0, ["th", "ought"])])
        assert types_of(events)[2:] == [
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
        ]

    @pytest.mark.asyncio
    async def test_signature_deltas_are_joined_into_encrypted_content(self):
        """Signature arrives in fragments and has no Responses event of its own;
        losing it makes the next turn's thinking block unreplayable."""
        events = await collect(
            [MESSAGE_START, *thinking_block(0, ["t"], signature_chunks=["sig-", "abc"])]
        )
        done = next(e for e in events if e.type == "response.output_item.done")
        assert done.item.encrypted_content == "sig-abc"

    @pytest.mark.asyncio
    async def test_no_event_is_emitted_for_a_signature_delta(self):
        events = await collect(
            [MESSAGE_START, *thinking_block(0, ["t"], signature_chunks=["a", "b", "c"])]
        )
        deltas = [
            e for e in events if e.type == "response.reasoning_summary_text.delta"
        ]
        assert len(deltas) == 1

    @pytest.mark.asyncio
    async def test_signature_survives_into_the_completed_response(self):
        events = await collect(
            [
                MESSAGE_START,
                *thinking_block(0, ["t"], signature_chunks=["sig"]),
                message_delta(),
                {"type": "message_stop"},
            ]
        )
        assert events[-1].response.output[0].encrypted_content == "sig"


class TestFunctionToolCalls:
    @pytest.mark.asyncio
    async def test_function_call_event_sequence(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "wait", ['{"s"', ": 3}"])],
            tools=[FUNCTION_TOOL],
        )
        assert types_of(events)[2:] == [
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]

    @pytest.mark.asyncio
    async def test_partial_json_is_forwarded_verbatim(self):
        """The client concatenates these fragments; re-encoding each one would
        produce invalid JSON on the other side."""
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "wait", ['{"s"', ": 3}"])],
            tools=[FUNCTION_TOOL],
        )
        deltas = [
            e.delta
            for e in events
            if e.type == "response.function_call_arguments.delta"
        ]
        assert deltas == ['{"s"', ": 3}"]

    @pytest.mark.asyncio
    async def test_done_event_carries_the_parsed_arguments(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "wait", ['{"s"', ": 3}"])],
            tools=[FUNCTION_TOOL],
        )
        done = next(
            e for e in events if e.type == "response.function_call_arguments.done"
        )
        assert json.loads(done.arguments) == {"s": 3}

    @pytest.mark.asyncio
    async def test_namespaced_call_restores_name_and_namespace(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "collab__spawn", ["{}"])],
            tools=[NAMESPACE_TOOL],
        )
        item = events[-1].item
        assert item.name == "spawn"
        assert item.namespace == "collab"

    @pytest.mark.asyncio
    async def test_tool_use_with_no_deltas_still_completes(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "wait", [])], tools=[FUNCTION_TOOL]
        )
        assert types_of(events)[-1] == "response.output_item.done"
        assert events[-1].item.arguments == "{}"


class TestCustomToolCalls:
    CHUNKS = ['{"input":', ' "echo ', '\\"hi', '\\" $V"}']

    @pytest.mark.asyncio
    async def test_custom_tool_emits_no_delta_until_the_block_closes(self):
        """Each fragment can split mid-escape, so nothing decodable exists until
        the whole JSON object has arrived."""
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "exec", self.CHUNKS)],
            tools=[CUSTOM_TOOL],
        )
        assert types_of(events)[2:] == [
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_item.done",
        ]

    @pytest.mark.asyncio
    async def test_input_split_at_an_escape_boundary_round_trips(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "exec", self.CHUNKS)],
            tools=[CUSTOM_TOOL],
        )
        done = next(
            e for e in events if e.type == "response.custom_tool_call_input.done"
        )
        assert done.input == 'echo "hi" $V'

    @pytest.mark.asyncio
    async def test_unicode_split_across_fragments_round_trips(self):
        events = await collect(
            [
                MESSAGE_START,
                *tool_use_block(0, "exec", ['{"input": "\\u4f', '60\\u597d"}']),
            ],
            tools=[CUSTOM_TOOL],
        )
        done = next(
            e for e in events if e.type == "response.custom_tool_call_input.done"
        )
        assert done.input == "你好"

    @pytest.mark.asyncio
    async def test_unparseable_input_falls_back_to_the_raw_buffer(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "exec", ['{"input": "trunc'])],
            tools=[CUSTOM_TOOL],
        )
        done = next(
            e for e in events if e.type == "response.custom_tool_call_input.done"
        )
        assert done.input == '{"input": "trunc'

    @pytest.mark.asyncio
    async def test_no_function_call_events_are_emitted_for_a_custom_tool(self):
        events = await collect(
            [MESSAGE_START, *tool_use_block(0, "exec", self.CHUNKS)],
            tools=[CUSTOM_TOOL],
        )
        assert not [e for e in events if "function_call_arguments" in str(e.type)]

    @pytest.mark.asyncio
    async def test_final_item_is_a_custom_tool_call(self):
        events = await collect(
            [
                MESSAGE_START,
                *tool_use_block(0, "exec", self.CHUNKS),
                message_delta(stop_reason="tool_use"),
                {"type": "message_stop"},
            ],
            tools=[CUSTOM_TOOL],
        )
        item = events[-1].response.output[0]
        assert item.type == "custom_tool_call"
        assert item.input == 'echo "hi" $V'


class TestMixedStream:
    CHUNKS = [
        MESSAGE_START,
        *thinking_block(0, ["let me check"], signature_chunks=["sig"]),
        *text_block(1, ["I will run it."]),
        *tool_use_block(2, "exec", ['{"input": "ls"}']),
        message_delta(stop_reason="tool_use"),
        {"type": "message_stop"},
    ]

    @pytest.mark.asyncio
    async def test_every_added_item_has_a_matching_done(self):
        """An unclosed item leaves the client's stream state hanging."""
        events = await collect(self.CHUNKS, tools=[CUSTOM_TOOL])
        added = [
            e.output_index for e in events if e.type == "response.output_item.added"
        ]
        done = [e.output_index for e in events if e.type == "response.output_item.done"]
        assert added == done == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_block_order_is_preserved_in_the_final_output(self):
        events = await collect(self.CHUNKS, tools=[CUSTOM_TOOL])
        assert [item.type for item in events[-1].response.output] == [
            "reasoning",
            "message",
            "custom_tool_call",
        ]

    @pytest.mark.asyncio
    async def test_streamed_output_matches_the_non_streaming_translation(self):
        from litellm.responses.anthropic_messages_transformation.transformation import (
            translate_anthropic_messages_response_to_responses_api_response,
        )

        context = translate_tools([CUSTOM_TOOL], None).context
        streamed = (await collect(self.CHUNKS, tools=[CUSTOM_TOOL]))[-1].response
        batched = translate_anthropic_messages_response_to_responses_api_response(
            response={
                "id": "msg_1",
                "model": "claude-opus-5",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 12, "output_tokens": 7},
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "let me check",
                        "signature": "sig",
                    },
                    {"type": "text", "text": "I will run it."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "exec",
                        "input": {"input": "ls"},
                    },
                ],
            },
            tool_context=context,
            request={},
            created_at=1700000000,
        )
        assert [i.model_dump() for i in streamed.output] == [
            i.model_dump() for i in batched.output
        ]
        assert streamed.usage.model_dump() == batched.usage.model_dump()
        assert streamed.status == batched.status

    @pytest.mark.asyncio
    async def test_response_created_advertises_the_original_tools(self):
        grammar_tool = {
            "type": "custom",
            "name": "exec",
            "format": {"type": "grammar", "syntax": "lark", "definition": "start: X"},
        }
        events = await collect(
            [MESSAGE_START], tools=[grammar_tool], request={"tools": [grammar_tool]}
        )
        advertised = events[0].response.tools[0].model_dump(exclude_none=True)
        assert advertised["format"] == grammar_tool["format"]


class TestBlocksWithNoOutputItem:
    """
    An MCP server runs its tools server-side, so Anthropic reports them as
    ``mcp_tool_use``/``mcp_tool_result`` blocks that have no Responses
    equivalent. They still occupy content indexes, so anything keyed off a
    block's position has to skip them.
    """

    MCP_TOOL = {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://mcp.example.com/sse",
        "require_approval": "never",
    }
    CHUNKS = [
        MESSAGE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "mcp_tool_use",
                "id": "mcp_1",
                "name": "search_docs",
                "server_name": "docs",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"q": "litellm"}'},
        },
        {"type": "content_block_stop", "index": 0},
        *text_block(1, ["The docs say hi."]),
        message_delta(),
        {"type": "message_stop"},
    ]

    @pytest.mark.asyncio
    async def test_no_delta_is_emitted_for_a_block_with_no_item(self):
        """A delta naming an item that was never added leaves the client with
        arguments it cannot attach to anything."""
        events = await collect(self.CHUNKS, tools=[self.MCP_TOOL])
        added = {
            (e.output_index, e.item.id)
            for e in events
            if e.type == "response.output_item.added"
        }
        addressed = {
            (e.output_index, e.item_id)
            for e in events
            if e.type
            in ("response.function_call_arguments.delta", "response.output_text.delta")
        }
        assert addressed <= added

    @pytest.mark.asyncio
    async def test_output_index_addresses_the_final_output_array(self):
        events = await collect(self.CHUNKS, tools=[self.MCP_TOOL])
        output = events[-1].response.output
        added = {
            e.output_index: e.item.id
            for e in events
            if e.type == "response.output_item.added"
        }
        assert added == {index: item.id for index, item in enumerate(output)}
