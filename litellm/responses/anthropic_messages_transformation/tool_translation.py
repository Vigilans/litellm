"""
Translates Responses API tool definitions into Anthropic ``/v1/messages`` tools.

Anthropic's tool schema is a flat list of JSON-schema tools, so ``custom`` tools
(free-form / grammar-constrained input) and ``namespace`` tools (one level of
nesting) have no direct equivalent and are lowered here. ``ToolTranslationContext``
records how each tool was lowered so the response side can restore the original
Responses shape.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from typing_extensions import Literal

from litellm._logging import verbose_logger
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicMcpServerTool,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
)
from litellm.types.llms.openai import (
    OpenAIMcpServerTool,
    OpenAIWebSearchOptions,
    OpenAIWebSearchUserLocation,
    ResponseInputParam,
)

ResponsesToolType = Literal["function", "custom"]

_ADDITIONAL_TOOLS_INPUT_ITEM_TYPE = "additional_tools"
_CUSTOM_TOOL_INPUT_FIELD = "input"
_NAMESPACE_SEPARATOR = "__"
_ANTHROPIC_TOOL_NAME_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_ANTHROPIC_TOOL_NAME_MAX_LEN = 64
_ANTHROPIC_TOOL_NAME_HASH_LEN = 12


@dataclass(frozen=True)
class ToolIdentity:
    anthropic_name: str
    response_name: str
    response_type: ResponsesToolType
    namespace: Optional[str] = None


@dataclass
class ToolTranslationContext:
    """
    Rebuilt from the request's tools on every turn; never cached across requests.
    """

    by_anthropic_name: Dict[str, ToolIdentity] = field(default_factory=dict)
    by_response_key: Dict[Tuple[Optional[str], str, str], ToolIdentity] = field(
        default_factory=dict
    )

    def add(self, identity: ToolIdentity) -> None:
        self.by_anthropic_name[identity.anthropic_name] = identity
        self.by_response_key[
            (identity.namespace, identity.response_name, identity.response_type)
        ] = identity

    def lookup_by_anthropic_name(self, name: str) -> Optional[ToolIdentity]:
        return self.by_anthropic_name.get(name)

    def resolve_response_tool(
        self,
        name: str,
        response_type: ResponsesToolType,
        namespace: Optional[str] = None,
    ) -> Optional[ToolIdentity]:
        """
        ``tool_choice`` has no namespace field, and a replayed ``function_call``
        carries one only optionally, so a name may arrive without its namespace.
        Fall back to the namespaced tools, but only when exactly one matches.
        """
        exact = self.by_response_key.get((namespace, name, response_type))
        if exact is not None or namespace is not None:
            return exact
        matches = [
            identity
            for (_, key_name, key_type), identity in self.by_response_key.items()
            if key_name == name and key_type == response_type
        ]
        return matches[0] if len(matches) == 1 else None


@dataclass
class ToolTranslationResult:
    tools: List[AllAnthropicToolsValues] = field(default_factory=list)
    mcp_servers: List[AnthropicMcpServerTool] = field(default_factory=list)
    context: ToolTranslationContext = field(default_factory=ToolTranslationContext)


def extract_additional_tools(
    input: Union[str, ResponseInputParam],
) -> Tuple[Union[str, ResponseInputParam], List[Dict[str, Any]]]:
    """
    Codex's "responses lite" wire mode ships tool definitions inside ``input`` as
    ``{"type": "additional_tools", "role": "developer", "tools": [...]}`` items
    rather than the top-level ``tools`` param. Pull those tools out and drop the
    items so they never reach the Anthropic message list.
    """
    if not isinstance(input, list):
        return input, []

    remaining: List[Any] = []
    hoisted: List[Dict[str, Any]] = []
    found_item = False
    for item in input:
        if _is_additional_tools_item(item):
            found_item = True
            tools = cast(Dict[str, Any], item).get("tools")
            if isinstance(tools, list):
                hoisted.extend(tools)
        else:
            remaining.append(item)

    if not found_item:
        return input, []

    verbose_logger.debug(
        "Responses -> Anthropic messages: hoisted %d tool(s) out of 'additional_tools' input item(s).",
        len(hoisted),
    )
    return cast(ResponseInputParam, remaining), hoisted


def translate_tools(
    top_level_tools: Optional[List[Any]],
    additional_tools: Optional[List[Any]],
) -> ToolTranslationResult:
    """
    Merge both places a Responses request can carry tools, then lower each one to
    Anthropic. Clients send either top-level ``tools`` (Claude Code) or
    ``additional_tools`` (Codex lite mode); read both so neither is dropped.
    """
    result = ToolTranslationResult()
    seen_keys: set = set()

    for tool in [*(top_level_tools or []), *(additional_tools or [])]:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "namespace":
            _translate_namespace_tool(tool, result, seen_keys)
        elif tool_type in ("function", "custom"):
            _translate_leaf_tool(
                tool, namespace=None, result=result, seen_keys=seen_keys
            )
        elif tool_type == "mcp":
            _translate_mcp_tool(tool, result)
        elif isinstance(tool_type, str) and tool_type.startswith("web_search"):
            result.tools.append(_translate_web_search_tool(tool))
        else:
            verbose_logger.debug(
                "Responses -> Anthropic messages: dropping unsupported tool type %s.",
                tool_type,
            )

    return result


def translate_tool_choice(
    tool_choice: Optional[Any],
    context: ToolTranslationContext,
    has_tools: bool,
) -> Optional[AnthropicMessagesToolChoice]:
    """
    A ``tool_choice`` naming a tool that was dropped, or any choice at all once no
    tool survived translation, must not be sent: Anthropic rejects a tool_choice
    with no matching tool.
    """
    if tool_choice is None or not has_tools:
        return None

    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return AnthropicMessagesToolChoice(type="auto")
        if tool_choice == "required":
            return AnthropicMessagesToolChoice(type="any")
        if tool_choice == "none":
            return AnthropicMessagesToolChoice(type="none")
        return None

    if not isinstance(tool_choice, dict):
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "allowed_tools":
        mode = tool_choice.get("mode")
        return AnthropicMessagesToolChoice(type="any" if mode == "required" else "auto")

    if choice_type in ("function", "custom"):
        name = tool_choice.get("name")
        identity = (
            context.resolve_response_tool(
                name=name, response_type=cast(ResponsesToolType, choice_type)
            )
            if isinstance(name, str)
            else None
        )
        if identity is None:
            verbose_logger.debug(
                "Responses -> Anthropic messages: dropping tool_choice for unmapped tool %s.",
                name,
            )
            return None
        choice = AnthropicMessagesToolChoice(type="tool")
        choice["name"] = identity.anthropic_name
        return choice

    verbose_logger.debug(
        "Responses -> Anthropic messages: dropping unsupported tool_choice type %s.",
        choice_type,
    )
    return None


def unwrap_custom_tool_input(tool_input: Any) -> str:
    """
    Restore the raw string a ``custom`` tool call carried. Only the exact shape
    ``translate_tools`` asks the model for is unwrapped; anything else is handed
    back as JSON so no model output is silently discarded.
    """
    if (
        isinstance(tool_input, dict)
        and set(tool_input.keys()) == {_CUSTOM_TOOL_INPUT_FIELD}
        and isinstance(tool_input[_CUSTOM_TOOL_INPUT_FIELD], str)
    ):
        return tool_input[_CUSTOM_TOOL_INPUT_FIELD]
    if isinstance(tool_input, str):
        return tool_input
    return json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False)


def wrap_custom_tool_input(raw_input: str) -> Dict[str, str]:
    return {_CUSTOM_TOOL_INPUT_FIELD: raw_input}


def _is_additional_tools_item(item: Any) -> bool:
    return (
        isinstance(item, dict) and item.get("type") == _ADDITIONAL_TOOLS_INPUT_ITEM_TYPE
    )


def _translate_namespace_tool(
    tool: Dict[str, Any],
    result: ToolTranslationResult,
    seen_keys: set,
) -> None:
    namespace = tool.get("name")
    if not isinstance(namespace, str) or not namespace:
        verbose_logger.debug(
            "Responses -> Anthropic messages: dropping namespace tool without a name."
        )
        return
    children = tool.get("tools")
    if not isinstance(children, list):
        return
    for child in children:
        if isinstance(child, dict) and child.get("type") in ("function", "custom"):
            _translate_leaf_tool(
                child,
                namespace=namespace,
                result=result,
                seen_keys=seen_keys,
                namespace_description=tool.get("description"),
            )


def _translate_leaf_tool(
    tool: Dict[str, Any],
    namespace: Optional[str],
    result: ToolTranslationResult,
    seen_keys: set,
    namespace_description: Optional[str] = None,
) -> None:
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        verbose_logger.debug(
            "Responses -> Anthropic messages: dropping tool without a name."
        )
        return

    response_type = cast(ResponsesToolType, tool["type"])
    key = (namespace, name, response_type)
    if key in seen_keys:
        return
    seen_keys.add(key)

    identity = ToolIdentity(
        anthropic_name=_resolve_anthropic_name(
            name=name,
            namespace=namespace,
            response_type=response_type,
            taken=result.context.by_anthropic_name,
        ),
        response_name=name,
        response_type=response_type,
        namespace=namespace,
    )
    result.context.add(identity)

    description = _build_description(tool, namespace, namespace_description)
    if response_type == "custom":
        result.tools.append(
            AnthropicMessagesTool(
                name=identity.anthropic_name,
                description=description,
                input_schema=cast(
                    Any,
                    {
                        "type": "object",
                        "properties": {
                            _CUSTOM_TOOL_INPUT_FIELD: {
                                "type": "string",
                                "description": "The complete tool input, passed through verbatim.",
                            }
                        },
                        "required": [_CUSTOM_TOOL_INPUT_FIELD],
                        "additionalProperties": False,
                    },
                ),
            )
        )
        return

    parameters = tool.get("parameters")
    input_schema: Dict[str, Any] = (
        dict(parameters) if isinstance(parameters, dict) else {}
    )
    input_schema.setdefault("type", "object")
    result.tools.append(
        AnthropicMessagesTool(
            name=identity.anthropic_name,
            description=description,
            input_schema=cast(Any, input_schema),
        )
    )


def _build_description(
    tool: Dict[str, Any],
    namespace: Optional[str],
    namespace_description: Optional[str],
) -> str:
    parts: List[str] = []
    if namespace:
        header = f"[namespace: {namespace}]"
        if namespace_description:
            header = f"{header} {namespace_description}"
        parts.append(header)

    description = tool.get("description")
    if isinstance(description, str) and description:
        parts.append(description)

    if tool.get("type") == "custom":
        parts.append(_describe_custom_format(tool.get("format")))

    return "\n\n".join(parts)


def _describe_custom_format(tool_format: Any) -> str:
    """
    Anthropic has no free-form or grammar-constrained tool input, so the format
    survives only as a description the model can read.
    """
    base = (
        f"This tool takes free-form text. Put the entire tool input in the "
        f"'{_CUSTOM_TOOL_INPUT_FIELD}' string field verbatim; do not wrap it in JSON."
    )
    if not isinstance(tool_format, dict) or tool_format.get("type") != "grammar":
        return base
    definition = tool_format.get("definition")
    if not isinstance(definition, str):
        return base
    return (
        f"{base}\n\nThe input must conform to the following "
        f"{tool_format.get('syntax')} grammar:\n{definition}"
    )


def _translate_mcp_tool(tool: Dict[str, Any], result: ToolTranslationResult) -> None:
    """Anthropic exposes remote MCP servers via the top-level ``mcp_servers`` param."""
    if not tool.get("server_url"):
        verbose_logger.debug(
            "Responses -> Anthropic messages: dropping mcp tool without a server_url "
            "(connector_id/tunnel_id have no Anthropic equivalent)."
        )
        return
    result.mcp_servers.append(
        AnthropicConfig()._map_openai_mcp_server_tool(cast(OpenAIMcpServerTool, tool))
    )


def _translate_web_search_tool(tool: Dict[str, Any]) -> AllAnthropicToolsValues:
    """
    ``AnthropicConfig.map_web_search_tool`` reads the chat-shaped
    ``user_location.approximate``; the Responses tool declares those fields flat,
    so re-nest them before delegating.
    """
    options = OpenAIWebSearchOptions(
        search_context_size=tool.get("search_context_size"),
    )
    user_location = tool.get("user_location")
    if isinstance(user_location, dict):
        options["user_location"] = cast(
            OpenAIWebSearchUserLocation,
            {
                "type": "approximate",
                "approximate": {
                    key: value
                    for key, value in user_location.items()
                    if key != "type" and value is not None
                },
            },
        )
    return AnthropicConfig().map_web_search_tool(options)


def _resolve_anthropic_name(
    name: str,
    namespace: Optional[str],
    response_type: ResponsesToolType,
    taken: Dict[str, ToolIdentity],
) -> str:
    """
    Names must stay stable across turns: a client may replay tool calls without
    resending the tool list, and this mapping is rebuilt from scratch on every
    request. So the disambiguating hash covers the full identity, never a list
    position.
    """
    flattened = f"{namespace}{_NAMESPACE_SEPARATOR}{name}" if namespace else name
    sanitized = _ANTHROPIC_TOOL_NAME_INVALID_CHARS.sub("_", flattened) or "tool"
    if len(sanitized) <= _ANTHROPIC_TOOL_NAME_MAX_LEN and sanitized not in taken:
        return sanitized

    digest = hashlib.sha256(
        json.dumps([namespace, name, response_type], separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:_ANTHROPIC_TOOL_NAME_HASH_LEN]
    keep = (
        _ANTHROPIC_TOOL_NAME_MAX_LEN
        - len(_NAMESPACE_SEPARATOR)
        - _ANTHROPIC_TOOL_NAME_HASH_LEN
    )
    return f"{sanitized[:keep]}{_NAMESPACE_SEPARATOR}{digest}"
