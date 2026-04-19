# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import Any, AsyncIterator, Dict

from litellm import verbose_logger
from litellm._uuid import uuid
from .utils import (
    build_text_blocks_with_citations,
    build_web_tool_use,
    build_web_search_results_from_annotations,
)


class AnthropicResponsesStreamWrapper:
    """
    Wraps a Responses API streaming iterator and re-emits events in Anthropic SSE format.

    Responses API event flow (relevant subset):
      response.created                        -> message_start
      response.output_item.added (message)    -> content_block_start (text)
      response.output_item.added (fn_call)    -> content_block_start (tool_use)
      response.output_item.added (reasoning)  -> (deferred to part.added)
      response.reasoning_summary_part.added   -> content_block_start (thinking)
      response.reasoning_summary_text.delta   -> content_block_delta (thinking_delta)
      response.reasoning_summary_part.done    -> content_block_stop
      response.output_text.delta              -> content_block_delta (text_delta)
      response.function_call_arguments.delta  -> content_block_delta (input_json_delta)
      response.output_item.done              -> content_block_stop (non-reasoning only)
      response.completed                     -> message_delta + message_stop

    Each reasoning summary part becomes its own thinking content block.
    This causes clients (e.g. Claude Code) to display each summary segment
    immediately upon its content_block_stop, rather than buffering all
    thinking until the entire reasoning phase completes.
    """

    def __init__(
        self,
        responses_stream: Any,
        model: str,
    ) -> None:
        self.responses_stream = responses_stream
        self.model = model
        self._message_id: str = f"msg_{uuid.uuid4()}"
        self._current_block_index: int = -1
        # Map output_index -> content_block_index for non-reasoning blocks.
        # Uses output_index instead of item_id because Copilot encrypts
        # item_ids differently per event, making them unreliable as keys.
        self._output_idx_to_block_index: Dict[int, int] = {}
        # Track open function_call items by item_id so we can emit tool_use start
        self._pending_tool_ids: Dict[str, str] = (
            {}
        )  # item_id -> call_id / name accumulator
        self._sent_message_start = False
        self._sent_message_stop = False
        self._chunk_queue: deque = deque()
        # Web tools: defer text emission until output_item.done
        self._web_tool_uses: list = []

    def _make_message_start(self) -> Dict[str, Any]:
        return {
            "type": "message_start",
            "message": {
                "id": self._message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }

    def _next_block_index(self) -> int:
        self._current_block_index += 1
        return self._current_block_index

    def _process_event(self, event: Any) -> None:  # noqa: PLR0915
        """Convert one Responses API event into zero or more Anthropic chunks queued for emission."""
        event_type = getattr(event, "type", None)
        if event_type is None and isinstance(event, dict):
            event_type = event.get("type")

        if event_type is None:
            return

        # ---- message_start ----
        if event_type == "response.created":
            if not self._sent_message_start:
                self._sent_message_start = True
                self._chunk_queue.append(self._make_message_start())
            return

        # ---- content_block_start for a new output message item ----
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None) or (
                event.get("item") if isinstance(event, dict) else None
            )
            if item is None:
                return
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            output_index = getattr(event, "output_index", None)
            if output_index is None and isinstance(event, dict):
                output_index = event.get("output_index")

            if item_type == "message":
                if self._web_tool_uses:
                    # Don't open text block here — deferred to
                    # output_item.done where full annotations (citations) are available.
                    return
                block_idx = self._next_block_index()
                if output_index is not None:
                    self._output_idx_to_block_index[output_index] = block_idx
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
            elif item_type == "function_call":
                item_id = getattr(item, "id", None) or (
                    item.get("id") if isinstance(item, dict) else None
                )
                call_id = (
                    getattr(item, "call_id", None)
                    or (item.get("call_id") if isinstance(item, dict) else None)
                    or ""
                )
                name = (
                    getattr(item, "name", None)
                    or (item.get("name") if isinstance(item, dict) else None)
                    or ""
                )
                block_idx = self._next_block_index()
                if output_index is not None:
                    self._output_idx_to_block_index[output_index] = block_idx
                if item_id:
                    self._pending_tool_ids[item_id] = call_id
                self._chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": {},
                        },
                    }
                )
            elif item_type == "web_search_call":
                # Don't emit content_block_start here — search queries
                # are only available in output_item.done.
                pass
            elif item_type == "reasoning":
                # Don't emit content_block_start here — each summary part
                # will open its own thinking block via part.added.
                pass
            return

        # ---- text delta ----
        if event_type == "response.output_text.delta":
            if self._web_tool_uses:
                # Don't stream text deltas here — full text with citation
                # is emitted from output_item.done.
                return
            output_index = getattr(event, "output_index", None)
            if output_index is None and isinstance(event, dict):
                output_index = event.get("output_index")
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            block_idx = (
                self._output_idx_to_block_index.get(output_index, self._current_block_index)
                if output_index is not None
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": delta},
                }
            )
            return

        # ---- reasoning summary part start -> new thinking content block ----
        # Each summary part becomes its own thinking block so that clients
        # display each segment immediately on content_block_stop.
        # Event flow per summary part:
        #   part.added (once)  ->  text.delta (many)  ->  part.done (once)
        if event_type == "response.reasoning_summary_part.added":
            block_idx = self._next_block_index()
            self._chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            )
            return

        # ---- reasoning summary text delta ----
        if event_type == "response.reasoning_summary_text.delta":
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": self._current_block_index,
                    "delta": {"type": "thinking_delta", "thinking": delta},
                }
            )
            return

        # ---- function call arguments delta ----
        if event_type == "response.function_call_arguments.delta":
            output_index = getattr(event, "output_index", None)
            if output_index is None and isinstance(event, dict):
                output_index = event.get("output_index")
            delta = getattr(event, "delta", "") or (
                event.get("delta", "") if isinstance(event, dict) else ""
            )
            block_idx = (
                self._output_idx_to_block_index.get(output_index, self._current_block_index)
                if output_index is not None
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                }
            )
            return

        # ---- reasoning summary part done -> content_block_stop ----
        if event_type == "response.reasoning_summary_part.done":
            self._chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": self._current_block_index,
                }
            )
            return

        # ---- output item done -> content_block_stop ----
        if event_type == "response.output_item.done":
            item = getattr(event, "item", None) or (
                event.get("item") if isinstance(event, dict) else None
            )
            item_type = (
                getattr(item, "type", None)
                or (item.get("type") if isinstance(item, dict) else None)
                if item
                else None
            )
            if item_type == "reasoning":
                # Reasoning items are closed by individual part.done events
                return
            if item_type == "web_search_call":
                self._emit_web_tool_use(item)
                return
            if item_type == "message" and self._web_tool_uses:
                if not isinstance(item, dict):
                    item = item.model_dump()
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text", "")
                        annotations = part.get("annotations") or []
                        citations = self._emit_web_search_results(annotations)
                        self._emit_cited_text_blocks(text, citations)
                        break
                return
            output_index = getattr(event, "output_index", None)
            if output_index is None and isinstance(event, dict):
                output_index = event.get("output_index")
            block_idx = (
                self._output_idx_to_block_index.get(output_index, self._current_block_index)
                if output_index is not None
                else self._current_block_index
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_stop",
                    "index": block_idx,
                }
            )
            return

        # ---- response completed -> message_delta + message_stop ----
        if event_type in (
            "response.completed",
            "response.failed",
            "response.incomplete",
        ):
            response_obj = getattr(event, "response", None) or (
                event.get("response") if isinstance(event, dict) else None
            )
            stop_reason = "end_turn"
            input_tokens = 0
            output_tokens = 0
            cache_creation_tokens = 0
            cache_read_tokens = 0

            if response_obj is not None:
                status = getattr(response_obj, "status", None)
                if status == "incomplete":
                    stop_reason = "max_tokens"
                usage = getattr(response_obj, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    cache_creation_tokens = getattr(usage, "input_tokens_details", None)  # type: ignore[assignment]
                    cache_read_tokens = getattr(usage, "output_tokens_details", None)  # type: ignore[assignment]
                    # Prefer direct cache fields if present
                    cache_creation_tokens = int(
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                    cache_read_tokens = int(
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    )

            # Check if tool_use was in the output to override stop_reason
            if response_obj is not None:
                output = getattr(response_obj, "output", []) or []
                for out_item in output:
                    out_type = getattr(out_item, "type", None) or (
                        out_item.get("type") if isinstance(out_item, dict) else None
                    )
                    if out_type == "function_call":
                        stop_reason = "tool_use"
                        break

            usage_delta: Dict[str, Any] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if cache_creation_tokens:
                usage_delta["cache_creation_input_tokens"] = cache_creation_tokens
            if cache_read_tokens:
                usage_delta["cache_read_input_tokens"] = cache_read_tokens
            if self._web_tool_uses:
                usage_delta["server_tool_use"] = {
                    "web_search_requests": sum(c["name"] == "web_search" for c in self._web_tool_uses),
                    "web_fetch_requests": sum(c["name"] == "web_fetch" for c in self._web_tool_uses),
                }

            self._chunk_queue.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage_delta,
                }
            )
            self._chunk_queue.append({"type": "message_stop"})
            self._sent_message_stop = True
            return

    def _emit_web_tool_use(self, item: Any) -> None:
        """Emit server_tool_use block and collect web tool info."""
        block, input_dict = build_web_tool_use(item)
        block_idx = self._next_block_index()
        self._web_tool_uses.append(block)
        self._chunk_queue.append(
            {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": block,
            }
        )
        self._chunk_queue.append(
            {
                "type": "content_block_delta",
                "index": block_idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(input_dict),
                },
            }
        )
        self._chunk_queue.append(
            {"type": "content_block_stop", "index": block_idx}
        )

    def _emit_web_search_results(self, annotations: list) -> list:
        """Emit web_search_tool_result blocks and return citations for text emission."""
        blocks, citations = build_web_search_results_from_annotations(
            self._web_tool_uses, annotations
        )
        for block in blocks:
            block_idx = self._next_block_index()
            self._chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": block,
                }
            )
            self._chunk_queue.append(
            {"type": "content_block_stop", "index": block_idx}
        )
        return citations

    def _emit_cited_text_blocks(self, text: str, citations: list) -> None:
        """Emit text blocks with citation deltas."""
        for block_data in build_text_blocks_with_citations(text, citations):
            block_idx = self._next_block_index()
            self._chunk_queue.append(
                {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            self._chunk_queue.append(
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": block_data["text"]},
                }
            )
            for cit in block_data.get("citations", []):
                self._chunk_queue.append(
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "citations_delta", "citation": cit},
                    }
                )
            self._chunk_queue.append(
            {"type": "content_block_stop", "index": block_idx}
        )

    def __aiter__(self) -> "AnthropicResponsesStreamWrapper":
        return self

    async def __anext__(self) -> Dict[str, Any]:
        # Return any queued chunks first
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        # Emit message_start if not yet done (fallback if response.created wasn't fired)
        if not self._sent_message_start:
            self._sent_message_start = True
            self._chunk_queue.append(self._make_message_start())
            return self._chunk_queue.popleft()

        # Consume the upstream stream
        try:
            async for event in self.responses_stream:
                self._process_event(event)
                if self._chunk_queue:
                    return self._chunk_queue.popleft()
        except StopAsyncIteration:
            pass
        except Exception as e:
            verbose_logger.error(
                f"AnthropicResponsesStreamWrapper error: {e}\n{traceback.format_exc()}"
            )

        # Drain any remaining queued chunks
        if self._chunk_queue:
            return self._chunk_queue.popleft()

        raise StopAsyncIteration

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """Yield SSE-encoded bytes for each Anthropic event chunk."""
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                yield chunk
