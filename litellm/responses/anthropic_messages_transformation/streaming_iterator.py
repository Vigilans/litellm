"""
Re-emits an Anthropic ``/v1/messages`` SSE stream as Responses API events.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from openai._streaming import SSEDecoder

from litellm._logging import verbose_logger
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    ContentPartDonePartOutputText,
    CustomToolCallInputDeltaEvent,
    CustomToolCallInputDoneEvent,
    ErrorEvent,
    ErrorEventError,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ReasoningSummaryPartDoneEvent,
    ReasoningSummaryTextDeltaEvent,
    ReasoningSummaryTextDoneEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponsePartAddedEvent,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)

from .tool_translation import ToolTranslationContext, unwrap_custom_tool_input
from .transformation import (
    translate_anthropic_messages_response_to_responses_api_response,
    translate_content_block_to_output_item,
)


@dataclass
class _OpenBlock:
    output_index: int
    item_id: str
    block: Dict[str, Any]
    text: str = ""
    signature: str = ""
    partial_json: str = ""
    is_custom_tool: bool = False


@dataclass
class _MessageState:
    """Mirrors the Anthropic message as it arrives so the terminal event can
    carry the same response the non-streaming path would have produced."""

    id: str = ""
    model: Optional[str] = None
    content: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    def as_anthropic_response(self) -> AnthropicMessagesResponse:
        return AnthropicMessagesResponse(
            id=self.id,
            type="message",
            role="assistant",
            model=self.model,
            content=cast(Any, self.content),
            stop_reason=cast(Any, self.stop_reason),
            usage=cast(AnthropicUsage, self.usage),
        )


class AnthropicMessagesToResponsesStreamIterator:
    def __init__(
        self,
        anthropic_stream: AsyncIterator[bytes],
        model: str,
        tool_context: ToolTranslationContext,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        created_at: int,
        responses_tools: Optional[List[Any]] = None,
    ):
        self.model = model
        self.tool_context = tool_context
        self.responses_api_request = responses_api_request
        self.created_at = created_at
        self.responses_tools = responses_tools

        self._sse_stream = SSEDecoder().aiter_bytes(anthropic_stream)
        self._pending: List[BaseLiteLLMOpenAIResponseObject] = []
        self._sequence_number = 0
        self._next_output_index = 0
        self._open_blocks: Dict[int, _OpenBlock] = {}
        self._state = _MessageState(model=model)

    def __aiter__(self) -> "AnthropicMessagesToResponsesStreamIterator":
        return self

    async def __anext__(self) -> BaseLiteLLMOpenAIResponseObject:
        while not self._pending:
            sse = await self._sse_stream.__anext__()
            if not sse.data:
                continue
            try:
                chunk = json.loads(sse.data)
            except json.JSONDecodeError:
                verbose_logger.debug(
                    "Anthropic messages -> Responses: skipping non-JSON SSE payload."
                )
                continue
            self._pending.extend(self.translate_chunk(chunk))
        return self._pending.pop(0)

    def translate_chunk(
        self, chunk: Dict[str, Any]
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        chunk_type = chunk.get("type")
        if chunk_type == "message_start":
            return self._on_message_start(chunk)
        if chunk_type == "content_block_start":
            return self._on_content_block_start(chunk)
        if chunk_type == "content_block_delta":
            return self._on_content_block_delta(chunk)
        if chunk_type == "content_block_stop":
            return self._on_content_block_stop(chunk)
        if chunk_type == "message_delta":
            return self._on_message_delta(chunk)
        if chunk_type == "message_stop":
            return self._on_message_stop()
        if chunk_type == "error":
            return self._on_error(chunk)
        return []

    def _next_sequence(self) -> int:
        self._sequence_number += 1
        return self._sequence_number

    def _snapshot(self, status: str) -> ResponsesAPIResponse:
        response = translate_anthropic_messages_response_to_responses_api_response(
            response=self._state.as_anthropic_response(),
            tool_context=self.tool_context,
            request=self.responses_api_request,
            created_at=self.created_at,
            responses_tools=self.responses_tools,
        )
        response.status = status
        return response

    def _on_message_start(
        self, chunk: Dict[str, Any]
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        message = chunk.get("message") or {}
        self._state.id = message.get("id") or ""
        self._state.model = message.get("model") or self.model
        self._state.usage = dict(message.get("usage") or {})
        return [
            ResponseCreatedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
                response=self._snapshot("in_progress"),
                sequence_number=self._next_sequence(),
            ),
            ResponseInProgressEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
                response=self._snapshot("in_progress"),
                sequence_number=self._next_sequence(),
            ),
        ]

    def _on_content_block_start(
        self, chunk: Dict[str, Any]
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        block = dict(chunk.get("content_block") or {})
        index = chunk.get("index", 0)
        item_id = f"{self._state.id}_{index}"

        item = translate_content_block_to_output_item(
            block=block, item_id=item_id, tool_context=self.tool_context
        )
        if item is None:
            # Recorded now so the final content indexes, and the item ids derived
            # from them, still line up; an open block would only collect deltas
            # that have no item to address.
            self._state.content.append(block)
            return []

        output_index = self._next_output_index
        self._next_output_index += 1

        identity = (
            self.tool_context.lookup_by_anthropic_name(block.get("name") or "")
            if block.get("type") == "tool_use"
            else None
        )
        self._open_blocks[index] = _OpenBlock(
            output_index=output_index,
            item_id=item_id,
            block=block,
            is_custom_tool=identity is not None and identity.response_type == "custom",
        )

        events: List[BaseLiteLLMOpenAIResponseObject] = [
            OutputItemAddedEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=output_index,
                item=BaseLiteLLMOpenAIResponseObject(**item),
                sequence_number=self._next_sequence(),
            )
        ]
        if block.get("type") == "text":
            events.append(
                ContentPartAddedEvent(
                    type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=0,
                    part=BaseLiteLLMOpenAIResponseObject(
                        type="output_text", text="", annotations=[]
                    ),
                    sequence_number=self._next_sequence(),
                )
            )
        elif block.get("type") in ("thinking", "redacted_thinking"):
            events.append(
                ResponsePartAddedEvent(
                    type=ResponsesAPIStreamEvents.RESPONSE_PART_ADDED,
                    item_id=item_id,
                    output_index=output_index,
                    summary_index=0,
                    part={"type": "summary_text", "text": ""},
                    sequence_number=self._next_sequence(),
                )
            )
        return events

    def _on_content_block_delta(
        self, chunk: Dict[str, Any]
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        open_block = self._open_blocks.get(chunk.get("index", 0))
        if open_block is None:
            return []
        delta = chunk.get("delta") or {}
        delta_type = delta.get("type")

        if delta_type == "text_delta":
            text = delta.get("text") or ""
            open_block.text += text
            return [
                OutputTextDeltaEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    content_index=0,
                    delta=text,
                    sequence_number=self._next_sequence(),
                )
            ]

        if delta_type == "thinking_delta":
            text = delta.get("thinking") or ""
            open_block.text += text
            return [
                ReasoningSummaryTextDeltaEvent(
                    type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    summary_index=0,
                    delta=text,
                    sequence_number=self._next_sequence(),
                )
            ]

        if delta_type == "signature_delta":
            # Carried on the finished reasoning item's encrypted_content; there
            # is no Responses event for a partial signature.
            open_block.signature += delta.get("signature") or ""
            return []

        if delta_type == "input_json_delta":
            partial = delta.get("partial_json") or ""
            open_block.partial_json += partial
            if open_block.is_custom_tool:
                # A custom tool's input is a raw string inside the JSON object, so
                # nothing can be emitted until the object parses: fragments split
                # mid-escape or mid-key are not decodable on their own.
                return []
            return [
                FunctionCallArgumentsDeltaEvent(
                    type=ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    delta=partial,
                    sequence_number=self._next_sequence(),
                )
            ]

        return []

    def _on_content_block_stop(
        self, chunk: Dict[str, Any]
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        index = chunk.get("index", 0)
        open_block = self._open_blocks.pop(index, None)
        if open_block is None:
            return []

        block = self._finalize_block(open_block)
        self._state.content.append(block)

        item = translate_content_block_to_output_item(
            block=block, item_id=open_block.item_id, tool_context=self.tool_context
        )
        if item is None:
            return []

        events = self._close_block_events(open_block, block, item)
        events.append(
            OutputItemDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=open_block.output_index,
                item=BaseLiteLLMOpenAIResponseObject(**item),
                sequence_number=self._next_sequence(),
            )
        )
        return events

    def _finalize_block(self, open_block: _OpenBlock) -> Dict[str, Any]:
        block = dict(open_block.block)
        block_type = block.get("type")
        if block_type == "text":
            block["text"] = open_block.text
        elif block_type == "thinking":
            block["thinking"] = open_block.text
            if open_block.signature:
                block["signature"] = open_block.signature
        elif block_type == "tool_use":
            block["input"] = self._parse_tool_input(open_block)
        return block

    def _parse_tool_input(self, open_block: _OpenBlock) -> Any:
        if not open_block.partial_json:
            return open_block.block.get("input") or {}
        try:
            return json.loads(open_block.partial_json)
        except json.JSONDecodeError:
            verbose_logger.debug(
                "Anthropic messages -> Responses: tool_use input did not parse as "
                "JSON; forwarding the raw accumulated text."
            )
            return open_block.partial_json

    def _close_block_events(
        self,
        open_block: _OpenBlock,
        block: Dict[str, Any],
        item: Dict[str, Any],
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        block_type = block.get("type")

        if block_type == "text":
            return [
                OutputTextDoneEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    content_index=0,
                    text=open_block.text,
                    sequence_number=self._next_sequence(),
                ),
                ContentPartDoneEvent(
                    type=ResponsesAPIStreamEvents.CONTENT_PART_DONE,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    content_index=0,
                    part=ContentPartDonePartOutputText(
                        type="output_text",
                        text=open_block.text,
                        annotations=[],
                        logprobs=None,
                    ),
                    sequence_number=self._next_sequence(),
                ),
            ]

        if block_type in ("thinking", "redacted_thinking"):
            return [
                ReasoningSummaryTextDoneEvent(
                    type=ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    summary_index=0,
                    text=open_block.text,
                    sequence_number=self._next_sequence(),
                ),
                ReasoningSummaryPartDoneEvent(
                    type=ResponsesAPIStreamEvents.REASONING_SUMMARY_PART_DONE,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    summary_index=0,
                    part=BaseLiteLLMOpenAIResponseObject(
                        type="summary_text", text=open_block.text
                    ),
                    sequence_number=self._next_sequence(),
                ),
            ]

        if block_type == "tool_use" and open_block.is_custom_tool:
            tool_input = unwrap_custom_tool_input(block.get("input"))
            return [
                CustomToolCallInputDeltaEvent(
                    type=ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DELTA,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    delta=tool_input,
                    sequence_number=self._next_sequence(),
                ),
                CustomToolCallInputDoneEvent(
                    type=ResponsesAPIStreamEvents.CUSTOM_TOOL_CALL_INPUT_DONE,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    input=tool_input,
                    sequence_number=self._next_sequence(),
                ),
            ]

        if block_type == "tool_use":
            return [
                FunctionCallArgumentsDoneEvent(
                    type=ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
                    item_id=open_block.item_id,
                    output_index=open_block.output_index,
                    arguments=item.get("arguments") or "{}",
                    sequence_number=self._next_sequence(),
                )
            ]

        return []

    def _on_message_delta(
        self, chunk: Dict[str, Any]
    ) -> List[BaseLiteLLMOpenAIResponseObject]:
        delta = chunk.get("delta") or {}
        if "stop_reason" in delta:
            self._state.stop_reason = delta.get("stop_reason")
        self._state.usage.update(chunk.get("usage") or {})
        return []

    def _on_message_stop(self) -> List[BaseLiteLLMOpenAIResponseObject]:
        response = self._snapshot_final()
        if response.status == "incomplete":
            return [
                ResponseIncompleteEvent(
                    type=ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
                    response=response,
                    sequence_number=self._next_sequence(),
                )
            ]
        return [
            ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=response,
                sequence_number=self._next_sequence(),
            )
        ]

    def _snapshot_final(self) -> ResponsesAPIResponse:
        return translate_anthropic_messages_response_to_responses_api_response(
            response=self._state.as_anthropic_response(),
            tool_context=self.tool_context,
            request=self.responses_api_request,
            created_at=self.created_at,
            responses_tools=self.responses_tools,
        )

    def _on_error(self, chunk: Dict[str, Any]) -> List[BaseLiteLLMOpenAIResponseObject]:
        error = chunk.get("error") or {}
        return [
            ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=self._next_sequence(),
                error=ErrorEventError(
                    type=error.get("type") or "api_error",
                    code=error.get("type") or "api_error",
                    message=error.get("message") or "",
                ),
            )
        ]
