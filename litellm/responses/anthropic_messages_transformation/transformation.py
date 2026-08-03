"""
Translates between the Responses API and Anthropic ``/v1/messages``.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.llms.anthropic import (
    AnthropicMessagesRequest,
    AnthropicOutputConfig,
    AnthropicOutputSchema,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStatus,
)

from .tool_translation import (
    ToolTranslationContext,
    extract_additional_tools,
    translate_tool_choice,
    translate_tools,
    unwrap_custom_tool_input,
    wrap_custom_tool_input,
)


@dataclass
class TranslatedMessagesRequest:
    request: AnthropicMessagesRequest
    tool_context: ToolTranslationContext
    responses_tools: List[Any]
    """Every tool the client sent, in Responses shape, for the response to echo."""


def translate_responses_request_to_anthropic_messages_request(
    model: str,
    input: Union[str, ResponseInputParam],
    responses_api_request: ResponsesAPIOptionalRequestParams,
) -> TranslatedMessagesRequest:
    remaining_input, additional_tools = extract_additional_tools(input)

    top_level_tools = cast(Optional[List[Any]], responses_api_request.get("tools"))
    tool_result = translate_tools(
        top_level_tools=top_level_tools,
        additional_tools=additional_tools,
    )
    system_blocks, messages = _translate_input(remaining_input, tool_result.context)

    instructions = responses_api_request.get("instructions")
    if instructions:
        system_blocks.insert(
            0, AnthropicSystemMessageContent(type="text", text=instructions)
        )

    request = AnthropicMessagesRequest(
        model=model,
        messages=cast(Any, messages),
        max_tokens=responses_api_request.get("max_output_tokens")
        or AnthropicConfig.get_max_tokens_for_model(model),
    )
    if system_blocks:
        request["system"] = cast(Any, system_blocks)
    if tool_result.tools:
        request["tools"] = cast(Any, tool_result.tools)
    if tool_result.mcp_servers:
        request["mcp_servers"] = tool_result.mcp_servers

    tool_choice = translate_tool_choice(
        tool_choice=responses_api_request.get("tool_choice"),
        context=tool_result.context,
        has_tools=bool(tool_result.tools),
    )
    if tool_choice is not None:
        parallel_tool_calls = responses_api_request.get("parallel_tool_calls")
        if parallel_tool_calls is not None and tool_choice["type"] != "none":
            tool_choice["disable_parallel_tool_use"] = not parallel_tool_calls
        request["tool_choice"] = tool_choice

    _map_optional_params(request, responses_api_request)

    return TranslatedMessagesRequest(
        request=request,
        tool_context=tool_result.context,
        responses_tools=[*(top_level_tools or []), *additional_tools],
    )


def _map_optional_params(
    request: AnthropicMessagesRequest,
    responses_api_request: ResponsesAPIOptionalRequestParams,
) -> None:
    temperature = responses_api_request.get("temperature")
    if temperature is not None:
        request["temperature"] = temperature
    top_p = responses_api_request.get("top_p")
    if top_p is not None:
        request["top_p"] = top_p
    stream = responses_api_request.get("stream")
    if stream is not None:
        request["stream"] = stream

    reasoning = responses_api_request.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str):
            # AnthropicMessagesConfig maps this to thinking/output_config without a
            # capability gate, so unmapped Claude-compatible models work too.
            request["reasoning_effort"] = effort

    output_schema = _extract_output_schema(responses_api_request.get("text"))
    if output_schema is not None:
        request["output_config"] = AnthropicOutputConfig(format=output_schema)


def _extract_output_schema(text_param: Any) -> Optional[AnthropicOutputSchema]:
    if not isinstance(text_param, dict):
        return None
    text_format = text_param.get("format")
    if not isinstance(text_format, dict) or text_format.get("type") != "json_schema":
        return None
    schema = text_format.get("schema")
    if not isinstance(schema, dict):
        return None
    return AnthropicOutputSchema(type="json_schema", schema=schema)


def _translate_input(
    input: Union[str, ResponseInputParam],
    context: ToolTranslationContext,
) -> Tuple[List[AnthropicSystemMessageContent], List[Dict[str, Any]]]:
    if isinstance(input, str):
        return [], [{"role": "user", "content": [{"type": "text", "text": input}]}]

    system_blocks: List[AnthropicSystemMessageContent] = []
    messages: List[Dict[str, Any]] = []
    for item in input:
        if not isinstance(item, dict):
            continue
        _translate_input_item(
            cast(Dict[str, Any], item), system_blocks, messages, context
        )
    return system_blocks, messages


def _translate_input_item(
    item: Dict[str, Any],
    system_blocks: List[AnthropicSystemMessageContent],
    messages: List[Dict[str, Any]],
    context: ToolTranslationContext,
) -> None:
    item_type = item.get("type", "message")

    if item_type in ("message", "easy_input_message") or (
        "role" in item and "content" in item
    ):
        role = item.get("role")
        if role in ("system", "developer"):
            system_blocks.extend(_translate_system_content(item.get("content")))
            return
        blocks = _translate_message_content(item.get("content"))
        if blocks:
            _append_blocks(
                messages, "assistant" if role == "assistant" else "user", blocks
            )
        return

    if item_type == "function_call":
        _append_blocks(messages, "assistant", [_translate_function_call(item, context)])
        return

    if item_type == "custom_tool_call":
        _append_blocks(
            messages, "assistant", [_translate_custom_tool_call(item, context)]
        )
        return

    if item_type in ("function_call_output", "custom_tool_call_output"):
        _append_blocks(messages, "user", [_translate_tool_call_output(item)])
        return

    if item_type == "reasoning":
        block = _translate_reasoning_item(item)
        if block is not None:
            _append_blocks(messages, "assistant", [block])
        return

    verbose_logger.debug(
        "Responses -> Anthropic messages: dropping unsupported input item type %s.",
        item_type,
    )


def _append_blocks(
    messages: List[Dict[str, Any]], role: str, blocks: List[Dict[str, Any]]
) -> None:
    """Anthropic requires alternating roles, so consecutive same-role items merge."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def _translate_system_content(content: Any) -> List[AnthropicSystemMessageContent]:
    blocks: List[AnthropicSystemMessageContent] = []
    if isinstance(content, str):
        if content:
            blocks.append(AnthropicSystemMessageContent(type="text", text=content))
        return blocks
    if not isinstance(content, list):
        return blocks
    for part in content:
        if isinstance(part, dict) and part.get("type") in (
            "input_text",
            "output_text",
            "text",
            "summary_text",
        ):
            text = part.get("text")
            if text:
                blocks.append(AnthropicSystemMessageContent(type="text", text=text))
    return blocks


def _translate_message_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []

    blocks: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text", "summary_text"):
            text = part.get("text")
            if text:
                blocks.append({"type": "text", "text": text})
        elif part_type == "input_image":
            block = _translate_input_image(part)
            if block is not None:
                blocks.append(block)
        elif part_type == "input_file":
            block = _translate_input_file(part)
            if block is not None:
                blocks.append(block)
        else:
            verbose_logger.debug(
                "Responses -> Anthropic messages: dropping unsupported content part %s.",
                part_type,
            )
    return blocks


def _translate_input_image(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from litellm.litellm_core_utils.prompt_templates.factory import (
        create_anthropic_image_param,
    )

    image_url = part.get("image_url")
    if isinstance(image_url, str) and image_url:
        return cast(Dict[str, Any], create_anthropic_image_param(image_url))
    file_id = part.get("file_id")
    if isinstance(file_id, str) and file_id:
        return {"type": "image", "source": {"type": "file", "file_id": file_id}}
    return None


def _translate_input_file(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    file_id = part.get("file_id")
    if isinstance(file_id, str) and file_id:
        return {"type": "document", "source": {"type": "file", "file_id": file_id}}
    file_url = part.get("file_url")
    if isinstance(file_url, str) and file_url:
        return {"type": "document", "source": {"type": "url", "url": file_url}}
    verbose_logger.debug(
        "Responses -> Anthropic messages: dropping input_file without file_id/file_url."
    )
    return None


def _translate_function_call(
    item: Dict[str, Any], context: ToolTranslationContext
) -> Dict[str, Any]:
    arguments = item.get("arguments")
    if isinstance(arguments, str):
        try:
            tool_input = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            raise litellm.exceptions.BadRequestError(
                message=(
                    f"function_call {item.get('call_id')!r} has arguments that are not "
                    f"valid JSON: {e}"
                ),
                model="",
                llm_provider="anthropic",
            )
    else:
        tool_input = arguments if isinstance(arguments, dict) else {}

    return {
        "type": "tool_use",
        "id": item.get("call_id"),
        "name": _resolve_replayed_tool_name(item, "function", context),
        "input": tool_input,
    }


def _translate_custom_tool_call(
    item: Dict[str, Any], context: ToolTranslationContext
) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "id": item.get("call_id"),
        "name": _resolve_replayed_tool_name(item, "custom", context),
        "input": wrap_custom_tool_input(item.get("input") or ""),
    }


def _resolve_replayed_tool_name(
    item: Dict[str, Any],
    response_type: str,
    context: ToolTranslationContext,
) -> Optional[str]:
    """
    A replayed call must name the same Anthropic tool the definition produced, or
    Claude sees a tool_use for a tool that isn't in ``tools``.
    """
    name = item.get("name")
    if not isinstance(name, str):
        return None
    identity = context.resolve_response_tool(
        name=name,
        response_type=cast(Any, response_type),
        namespace=item.get("namespace"),
    )
    return identity.anthropic_name if identity is not None else name


def _translate_tool_call_output(item: Dict[str, Any]) -> Dict[str, Any]:
    output = item.get("output")
    if not isinstance(output, str):
        output = json.dumps(output, separators=(",", ":"), ensure_ascii=False)
    return {
        "type": "tool_result",
        "tool_use_id": item.get("call_id"),
        "content": output,
    }


def _translate_reasoning_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Anthropic rejects a ``thinking`` block without its ``signature``, so a
    reasoning item that lost its ``encrypted_content`` (the client did not request
    it, or a different backend produced it) cannot be replayed.
    """
    signature = item.get("encrypted_content")
    if not isinstance(signature, str) or not signature:
        verbose_logger.debug(
            "Responses -> Anthropic messages: dropping reasoning item with no "
            "encrypted_content to restore the thinking signature from."
        )
        return None
    return {
        "type": "thinking",
        "thinking": _reasoning_item_text(item),
        "signature": signature,
    }


def _reasoning_item_text(item: Dict[str, Any]) -> str:
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, list):
            texts = [
                part["text"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if texts:
                return "\n\n".join(texts)
    return ""


def translate_anthropic_messages_response_to_responses_api_response(
    response: AnthropicMessagesResponse,
    tool_context: ToolTranslationContext,
    request: ResponsesAPIOptionalRequestParams,
    created_at: int,
    responses_tools: Optional[List[Any]] = None,
) -> ResponsesAPIResponse:
    output: List[Dict[str, Any]] = []
    for index, block in enumerate(response.get("content") or []):
        if not isinstance(block, dict):
            continue
        item = translate_content_block_to_output_item(
            block=cast(Dict[str, Any], block),
            item_id=f"{response.get('id')}_{index}",
            tool_context=tool_context,
        )
        if item is not None:
            output.append(item)

    stop_reason = response.get("stop_reason")
    status, incomplete_details = _map_stop_reason(stop_reason)
    return ResponsesAPIResponse(
        id=cast(str, response.get("id")),
        created_at=created_at,
        model=response.get("model"),
        object="response",
        output=cast(Any, output),
        status=status,
        incomplete_details=incomplete_details,
        usage=translate_usage(cast(Optional[Dict[str, Any]], response.get("usage"))),
        instructions=request.get("instructions"),
        metadata=request.get("metadata"),
        parallel_tool_calls=request.get("parallel_tool_calls"),
        temperature=request.get("temperature"),
        text=cast(Any, request.get("text")),
        # Advertise the tools the client sent, not the lowered Anthropic ones.
        # Codex ships them inside an `additional_tools` input item rather than
        # the top-level param, so fall back to that only when nothing was hoisted.
        tools=cast(
            Any,
            responses_tools if responses_tools is not None else request.get("tools"),
        ),
        tool_choice=request.get("tool_choice"),
        top_p=request.get("top_p"),
        max_output_tokens=request.get("max_output_tokens"),
        reasoning=cast(Any, request.get("reasoning")),
        truncation=request.get("truncation"),
        user=request.get("user"),
        store=request.get("store"),
    )


def translate_content_block_to_output_item(
    block: Dict[str, Any],
    item_id: str,
    tool_context: ToolTranslationContext,
) -> Optional[Dict[str, Any]]:
    """
    Anthropic emits thinking, text, and tool_use blocks in a single ordered list;
    each maps to one Responses output item and the order is preserved.
    """
    block_type = block.get("type")

    if block_type == "text":
        return {
            "type": "message",
            "id": item_id,
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": block.get("text") or "",
                    "annotations": [],
                }
            ],
        }

    if block_type == "thinking":
        return {
            "type": "reasoning",
            "id": item_id,
            "status": "completed",
            "summary": [{"type": "summary_text", "text": block.get("thinking") or ""}],
            # Anthropic requires the signature back verbatim on the next turn;
            # encrypted_content is the field the client replays.
            "encrypted_content": block.get("signature"),
        }

    if block_type == "redacted_thinking":
        return {
            "type": "reasoning",
            "id": item_id,
            "status": "completed",
            "summary": [],
            "encrypted_content": block.get("data"),
        }

    if block_type == "tool_use":
        return _translate_tool_use_block(block, item_id, tool_context)

    verbose_logger.debug(
        "Anthropic messages -> Responses: dropping unsupported content block %s.",
        block_type,
    )
    return None


def _translate_tool_use_block(
    block: Dict[str, Any],
    item_id: str,
    tool_context: ToolTranslationContext,
) -> Dict[str, Any]:
    name = block.get("name") or ""
    identity = tool_context.lookup_by_anthropic_name(name)
    tool_input = block.get("input")

    if identity is not None and identity.response_type == "custom":
        item: Dict[str, Any] = {
            "type": "custom_tool_call",
            "id": item_id,
            "call_id": block.get("id"),
            "name": identity.response_name,
            "input": unwrap_custom_tool_input(tool_input),
        }
    else:
        item = {
            "type": "function_call",
            "id": item_id,
            "status": "completed",
            "call_id": block.get("id"),
            "name": identity.response_name if identity is not None else name,
            "arguments": json.dumps(
                tool_input if tool_input is not None else {},
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        }

    if identity is not None and identity.namespace is not None:
        item["namespace"] = identity.namespace
    return item


def translate_usage(usage: Optional[Dict[str, Any]]) -> Optional[ResponseAPIUsage]:
    if not usage:
        return None
    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    return ResponseAPIUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=cache_read,
            cache_creation_tokens=cache_creation,
        ),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _map_stop_reason(
    stop_reason: Optional[str],
) -> Tuple[ResponsesAPIStatus, Optional[Dict[str, str]]]:
    if stop_reason in ("max_tokens", "model_context_window_exceeded"):
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None
