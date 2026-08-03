import copy
from enum import Enum
from typing import Any, Dict, Optional

from litellm._internal_context import is_web_search_call
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

from .transformation import (
    chat_to_search_response,
    messages_to_search_response,
    responses_to_search_response,
)


class SearchModelEndpoint(str, Enum):
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    CHAT = "chat"


def _as_metadata_copy(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    copied: Dict[str, Any] = {}
    for key, item in value.items():
        try:
            copied[key] = copy.deepcopy(item)
        except Exception:
            copied[key] = item
    return copied


def _merged_metadata(request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _as_metadata_copy(request_kwargs.get("metadata"))
    metadata.update(_as_metadata_copy(request_kwargs.get("litellm_metadata")))
    metadata.pop("user_api_key_budget_reservation", None)
    metadata.pop("user_api_key_auth", None)
    return metadata


def _provider_for_selection(
    *, model: str, custom_llm_provider: Optional[str]
) -> tuple[str, str]:
    if custom_llm_provider:
        return model, custom_llm_provider

    provider, separator, provider_model = model.partition("/")
    if separator and provider in LlmProviders._value2member_map_:
        return provider_model, provider
    return model, ""


def select_search_model_endpoint(
    *,
    model: str,
    custom_llm_provider: str,
    model_info: Optional[Dict[str, Any]],
) -> SearchModelEndpoint:
    if (model_info or {}).get("mode") == "responses":
        return SearchModelEndpoint.RESPONSES

    try:
        provider = LlmProviders(custom_llm_provider)
    except ValueError:
        return SearchModelEndpoint.CHAT

    if (
        ProviderConfigManager.get_provider_responses_api_config(
            model=model,
            provider=provider,
        )
        is not None
    ):
        return SearchModelEndpoint.RESPONSES
    if (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=provider,
        )
        is not None
    ):
        return SearchModelEndpoint.ANTHROPIC_MESSAGES
    return SearchModelEndpoint.CHAT


def _responses_search_tool(optional_params: Dict[str, Any]) -> Dict[str, Any]:
    tool: Dict[str, Any] = {"type": "web_search"}
    if optional_params.get("search_domain_filter"):
        tool["filters"] = {"allowed_domains": optional_params["search_domain_filter"]}
    if optional_params.get("country"):
        tool["user_location"] = {
            "type": "approximate",
            "country": optional_params["country"],
        }
    return tool


def _messages_search_tool(optional_params: Dict[str, Any]) -> Dict[str, Any]:
    tool: Dict[str, Any] = {
        "type": "web_search_20250305",
        "name": "web_search",
    }
    if optional_params.get("country"):
        tool["user_location"] = {
            "type": "approximate",
            "country": optional_params["country"],
        }
    return tool


def _chat_search_options(optional_params: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if optional_params.get("country"):
        options["user_location"] = {
            "type": "approximate",
            "approximate": {"country": optional_params["country"]},
        }
    return options


def _endpoint_call_kwargs(
    *,
    model: str,
    custom_llm_provider: str,
    api_key: Optional[str],
    api_base: Optional[str],
    timeout: Optional[Any],
    model_info: Optional[Dict[str, Any]],
    request_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    excluded_keys = {
        "caching",
        "disable_fallbacks",
        "litellm_call_id",
        "litellm_logging_obj",
        "litellm_metadata",
        "metadata",
        "input",
        "messages",
        "model_info",
        "num_retries",
        "query",
        "search_optional_params",
        "search_tool_name",
        "tools",
        "web_search_options",
    }
    call_kwargs = {
        key: value for key, value in request_kwargs.items() if key not in excluded_keys
    }
    metadata = _merged_metadata(request_kwargs)
    call_kwargs.update(
        {
            "model": model,
            "custom_llm_provider": custom_llm_provider,
            "stream": False,
            "metadata": _as_metadata_copy(metadata),
            "litellm_metadata": metadata,
            "model_info": _as_metadata_copy(model_info),
        }
    )
    if api_key is not None:
        call_kwargs["api_key"] = api_key
    if api_base is not None:
        call_kwargs["api_base"] = api_base
    if timeout is not None:
        call_kwargs["timeout"] = timeout
    return call_kwargs


async def _call_selected_search_model(
    *,
    query: str,
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Any] = None,
    search_optional_params: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> SearchResponse:
    import litellm

    endpoint_model, endpoint_provider = _provider_for_selection(
        model=model,
        custom_llm_provider=custom_llm_provider,
    )
    endpoint = select_search_model_endpoint(
        model=endpoint_model,
        custom_llm_provider=endpoint_provider,
        model_info=model_info,
    )
    call_kwargs = _endpoint_call_kwargs(
        model=model,
        custom_llm_provider=endpoint_provider,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        model_info=model_info,
        request_kwargs=kwargs,
    )
    optional_params = search_optional_params or {}

    token = is_web_search_call.set(True)
    try:
        if endpoint == SearchModelEndpoint.RESPONSES:
            response = await litellm.aresponses(
                input=query,
                tools=[_responses_search_tool(optional_params)],
                **call_kwargs,
            )
            return responses_to_search_response(response=response, query=query)
        if endpoint == SearchModelEndpoint.ANTHROPIC_MESSAGES:
            response = await litellm.anthropic_messages(
                max_tokens=call_kwargs.pop("max_tokens", 1024),
                messages=[{"role": "user", "content": query}],
                tools=[_messages_search_tool(optional_params)],
                **call_kwargs,
            )
            return messages_to_search_response(response=response, query=query)

        response = await litellm.acompletion(
            messages=[{"role": "user", "content": query}],
            web_search_options=_chat_search_options(optional_params),
            **call_kwargs,
        )
        return chat_to_search_response(response=response, query=query)
    finally:
        is_web_search_call.reset(token)


async def execute_model_search(
    *,
    query: str,
    model: str,
    timeout: Optional[Any] = None,
    optional_params: Optional[Dict[str, Any]] = None,
    request_kwargs: Optional[Dict[str, Any]] = None,
    num_retries: Optional[int] = None,
) -> SearchResponse:
    from litellm.search._context import search_router

    router = search_router.get()
    if router is None:
        raise ValueError(f"Search model '{model}' requires a Router")

    router_kwargs = dict(request_kwargs or {})
    router_kwargs.pop("litellm_logging_obj", None)
    router_kwargs.pop("litellm_call_id", None)
    for key in ("metadata", "litellm_metadata"):
        if key in router_kwargs:
            router_kwargs[key] = _as_metadata_copy(router_kwargs[key])
    router_kwargs.update(
        {
            "query": query,
            "search_optional_params": optional_params or {},
            "disable_fallbacks": True,
        }
    )
    if timeout is not None:
        router_kwargs["timeout"] = timeout
    if num_retries is not None:
        router_kwargs["num_retries"] = num_retries

    return await router._ageneric_api_call_with_fallbacks(
        model=model,
        original_function=_call_selected_search_model,
        **router_kwargs,
    )
