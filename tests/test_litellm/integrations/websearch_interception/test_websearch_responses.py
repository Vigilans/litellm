from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.responses.streaming_iterator import CachedResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CallTypes


def _response(*output):
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=1,
        model="gpt-5.6-sol",
        object="response",
        output=list(output),
        status="completed",
    )


def _request_data(**overrides):
    request = {
        "model": "gpt-5.6-sol",
        "input": [{"role": "user", "content": "latest news"}],
        "tools": [
            {
                "type": "function",
                "name": "litellm_web_search",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": "required",
        "custom_llm_provider": "github_copilot",
        "litellm_metadata": {"deployment_model_name": "gpt-5.6-sol-max"},
    }
    request.update(overrides)
    return request


@pytest.mark.asyncio
async def test_execute_search_uses_configured_router_search_tool():
    handler = WebSearchInterceptionLogger(search_tool_name="llm-search")
    router = MagicMock()
    router.search_tools = [{"search_tool_name": "llm-search"}]
    router.asearch = AsyncMock(return_value=SearchResponse(results=[]))

    with patch.dict(
        "sys.modules",
        {"litellm.proxy.proxy_server": MagicMock(llm_router=router)},
    ):
        await handler._execute_search("latest news")

    router.asearch.assert_awaited_once_with(
        query="latest news", search_tool_name="llm-search"
    )


@pytest.mark.asyncio
async def test_responses_hook_replays_output_and_preserves_search_citations():
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
    function_call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "litellm_web_search",
        "arguments": '{"query":"latest news"}',
    }
    initial = _response(reasoning, function_call)
    final = _response(
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Final answer.", "annotations": []}
            ],
        }
    )
    search_response = SearchResponse(
        results=[
            SearchResult(
                title="Example",
                url="https://example.com/source",
                snippet="Evidence",
            )
        ]
    )
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    handler._execute_search = AsyncMock(  # type: ignore[method-assign]
        return_value=("Title: Example", search_response)
    )

    with patch("litellm.aresponses", new=AsyncMock(return_value=final)) as call:
        result = await handler.async_post_call_success_deployment_hook(
            _request_data(), initial, CallTypes.aresponses
        )

    assert result is final
    follow_up = call.await_args.kwargs
    assert follow_up["model"] == "gpt-5.6-sol"
    assert follow_up["tools"] == _request_data()["tools"]
    assert follow_up["tool_choice"] == "auto"
    assert follow_up["input"] == [
        {"role": "user", "content": "latest news"},
        reasoning,
        function_call,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "Title: Example",
        },
    ]
    text = result.output[0]["content"][0]
    assert text["text"] == "Final answer. [Example](https://example.com/source)"
    assert [annotation.model_dump() for annotation in text["annotations"]] == [
        {
            "type": "url_citation",
            "url": "https://example.com/source",
            "title": "Example",
            "start_index": 13,
            "end_index": 51,
        }
    ]


@pytest.mark.asyncio
async def test_responses_hook_restores_requested_stream():
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    logging_obj.dynamic_success_callbacks = []
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    handler._execute_search = AsyncMock(  # type: ignore[method-assign]
        return_value=("result", SearchResponse(results=[]))
    )
    initial = _response(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "litellm_web_search",
            "arguments": '{"query":"latest news"}',
        }
    )

    with patch("litellm.aresponses", new=AsyncMock(return_value=_response())):
        result = await handler.async_post_call_success_deployment_hook(
            _request_data(
                _websearch_interception_converted_stream=True,
                litellm_logging_obj=logging_obj,
            ),
            initial,
            CallTypes.aresponses,
        )

    assert isinstance(result, CachedResponsesAPIStreamingIterator)


@pytest.mark.asyncio
async def test_responses_hook_uses_selected_router_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    response = _response(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "litellm_web_search",
            "arguments": '{"query":"latest news"}',
        }
    )

    result = await handler.async_post_call_success_deployment_hook(
        _request_data(litellm_metadata={"deployment_model_name": "gpt-5.6-sol"}),
        response,
        CallTypes.aresponses,
    )

    assert result is None


@pytest.mark.asyncio
async def test_responses_hook_rejects_repeated_tool_call():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    function_call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "litellm_web_search",
        "arguments": '{"query":"latest news"}',
    }
    response = _response(function_call)
    fingerprint = '[{"call_id": "call_1", "input": {"query": "latest news"}, "name": "litellm_web_search"}]'

    with pytest.raises(ValueError, match="repeated tool-call fingerprint"):
        await handler.async_post_call_success_deployment_hook(
            _request_data(_websearch_interception_responses_fingerprints=[fingerprint]),
            response,
            CallTypes.aresponses,
        )


@pytest.mark.asyncio
async def test_responses_hook_propagates_request_callbacks():
    callback = MagicMock()
    logging_obj = MagicMock()
    logging_obj.dynamic_success_callbacks = [callback]
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    handler._execute_search = AsyncMock(  # type: ignore[method-assign]
        return_value=("result", SearchResponse(results=[]))
    )
    response = _response(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "litellm_web_search",
            "arguments": '{"query":"latest news"}',
        }
    )

    with patch("litellm.aresponses", new=AsyncMock(return_value=_response())) as call:
        await handler.async_post_call_success_deployment_hook(
            _request_data(litellm_logging_obj=logging_obj),
            response,
            CallTypes.aresponses,
        )

    assert call.await_args.kwargs["success_callback"] == [callback]
