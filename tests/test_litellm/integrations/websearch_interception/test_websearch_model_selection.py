import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.utils import ModelResponse


@pytest.mark.parametrize(
    "enabled_models,model_name,expected",
    [
        (None, "anything", True),
        ([], "claude-opus-5-max", False),
        (["claude-*"], "claude-opus-5-max", True),
        (["gpt-5.6-sol-*"], "gpt-5.6-sol-max", True),
        (["*-max"], "gpt-5.6-terra-max", True),
        (["anthropic/claude-*"], "anthropic/claude-sonnet-4-6", True),
        (["claude-*"], "anthropic/claude-sonnet-4-6", False),
    ],
)
def test_enabled_models_uses_full_model_name_globs(
    enabled_models, model_name, expected
):
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=enabled_models
    )

    assert handler._model_is_enabled(model_name) is expected


def test_config_yaml_carries_enabled_models():
    handler = WebSearchInterceptionLogger.from_config_yaml(
        {
            "enabled_providers": ["github_copilot"],
            "enabled_models": ["claude-*", "*-max"],
            "search_tool_name": "llm-search",
        }
    )

    assert handler.enabled_models == ["claude-*", "*-max"]


@pytest.mark.asyncio
async def test_short_circuit_matches_selected_router_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["custom_provider"], enabled_models=["*-max"]
    )
    handler._execute_search = AsyncMock(return_value=("result", None))  # type: ignore[method-assign]

    result = await handler.try_short_circuit_search(
        model="backend-model",
        deployment_model_name="claude-opus-5-max",
        messages=[{"role": "user", "content": "query"}],
        tools=[{"type": "web_search"}],
        custom_llm_provider="custom_provider",
    )

    assert result is not None


@pytest.mark.asyncio
async def test_router_passes_selected_model_name_to_deployment_hook(monkeypatch):
    import litellm
    from litellm import Router

    handler = WebSearchInterceptionLogger(
        enabled_providers=["openai"], enabled_models=["*-max"]
    )
    monkeypatch.setattr(litellm, "callbacks", [handler])
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.6-sol-max",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-fake",
                    "mock_response": "ok",
                },
            }
        ]
    )
    seen = {}
    original = handler.async_pre_call_deployment_hook

    async def record(kwargs, call_type):
        seen["model"] = kwargs["model"]
        seen["model_name"] = kwargs["metadata"]["deployment_model_name"]
        return await original(kwargs, call_type)

    monkeypatch.setattr(handler, "async_pre_call_deployment_hook", record)

    await router.acompletion(
        model="gpt-5.6-sol-max",
        messages=[{"role": "user", "content": "query"}],
        tools=[{"type": "web_search"}],
    )

    assert seen == {
        "model": "openai/gpt-4o",
        "model_name": "gpt-5.6-sol-max",
    }


@pytest.mark.asyncio
async def test_router_passes_selected_model_name_to_short_circuit(monkeypatch):
    import litellm
    from litellm import Router

    handler = WebSearchInterceptionLogger(
        enabled_providers=["anthropic"], enabled_models=["anthropic/*-max"]
    )
    monkeypatch.setattr(litellm, "callbacks", [handler])
    router = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-sonnet-4-6-max",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "api_key": "sk-fake",
                    "mock_response": "ok",
                },
            }
        ]
    )
    seen = {}

    async def record(**kwargs):
        seen["model"] = kwargs["model"]
        seen["model_name"] = kwargs["deployment_model_name"]
        return None

    monkeypatch.setattr(handler, "try_short_circuit_search", record)

    await router.anthropic_messages(
        model="anthropic/claude-sonnet-4-6-max",
        max_tokens=16,
        messages=[{"role": "user", "content": "query"}],
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )

    assert seen == {
        "model": "anthropic/claude-sonnet-4-6",
        "model_name": "anthropic/claude-sonnet-4-6-max",
    }


@pytest.mark.asyncio
async def test_short_circuit_rejects_unmatched_selected_router_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["custom_provider"], enabled_models=["*-max"]
    )

    result = await handler.try_short_circuit_search(
        model="backend-model",
        deployment_model_name="claude-opus-5",
        messages=[{"role": "user", "content": "query"}],
        tools=[{"type": "web_search"}],
        custom_llm_provider="custom_provider",
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
async def test_deployment_hook_matches_selected_router_model_name(metadata_key):
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )

    kwargs = {
        "model": "github_copilot/claude-opus-5",
        "custom_llm_provider": "github_copilot",
        "tools": [{"type": "web_search"}],
        metadata_key: {"deployment_model_name": "claude-opus-5-max"},
    }
    result = await handler.async_pre_call_deployment_hook(kwargs, None)

    assert result is not None


@pytest.mark.asyncio
async def test_messages_pre_request_hook_matches_selected_router_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )

    result = await handler.async_pre_request_hook(
        model="github_copilot/claude-opus-5",
        messages=[],
        kwargs={
            "tools": [{"type": "web_search"}],
            "litellm_params": {"custom_llm_provider": "github_copilot"},
            "litellm_metadata": {"deployment_model_name": "claude-opus-5-max"},
        },
    )

    assert result is not None
    assert result["tools"][0]["name"] == "litellm_web_search"


@pytest.mark.asyncio
async def test_anthropic_agentic_hook_matches_selected_router_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["bedrock"], enabled_models=["*-max"]
    )
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "litellm_web_search",
                "input": {"query": "query"},
            }
        ]
    }

    should_run, _ = await handler.async_should_run_agentic_loop(
        response=response,
        model="anthropic.claude-opus-4-6-v1:0",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="bedrock",
        kwargs={"deployment_model_name": "anthropic/claude-opus-4-6-max"},
    )

    assert should_run is True


@pytest.mark.asyncio
async def test_messages_dispatcher_reads_selected_model_from_metadata(monkeypatch):
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    seen = {}

    async def record(**kwargs):
        seen["model_name"] = kwargs["kwargs"]["deployment_model_name"]
        return False, {}

    monkeypatch.setattr(handler, "async_should_run_agentic_loop", record)
    monkeypatch.setattr("litellm.callbacks", [handler])
    logging_obj = AsyncMock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {}

    await BaseLLMHTTPHandler()._call_agentic_completion_hooks(
        response={"content": []},
        model="claude-opus-5",
        messages=[],
        anthropic_messages_provider_config=AsyncMock(),
        anthropic_messages_optional_request_params={"tools": []},
        logging_obj=logging_obj,
        stream=False,
        custom_llm_provider="github_copilot",
        kwargs={"litellm_metadata": {"deployment_model_name": "claude-opus-5-max"}},
    )

    assert seen["model_name"] == "claude-opus-5-max"


@pytest.mark.asyncio
async def test_chat_agentic_hook_matches_selected_router_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    response = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "litellm_web_search",
                                "arguments": '{"query":"query"}',
                            },
                        }
                    ],
                },
            }
        ]
    )

    should_run, _ = await handler.async_should_run_chat_completion_agentic_loop(
        response=response,
        model="claude-opus-5",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="github_copilot",
        kwargs={"deployment_model_name": "claude-opus-5-max"},
    )

    assert should_run is True


@pytest.mark.asyncio
async def test_direct_anthropic_agentic_hook_keeps_qualified_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["anthropic"], enabled_models=["anthropic/claude-*"]
    )
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "litellm_web_search",
                "input": {"query": "query"},
            }
        ]
    }

    should_run, _ = await handler.async_should_run_agentic_loop(
        response=response,
        model="claude-sonnet-4-6",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_web_search"},
            }
        ],
        stream=False,
        custom_llm_provider="anthropic",
        kwargs={"deployment_model_name": "anthropic/claude-sonnet-4-6"},
    )

    assert should_run is True


@pytest.mark.asyncio
async def test_provider_and_model_filters_are_both_required():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["bedrock"], enabled_models=["claude-*"]
    )

    result = await handler.async_pre_call_deployment_hook(
        {
            "model": "github_copilot/claude-opus-5",
            "custom_llm_provider": "github_copilot",
            "tools": [{"type": "web_search"}],
            "litellm_metadata": {"deployment_model_name": "claude-opus-5"},
        },
        None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_rewritten_tool_matches_the_endpoint_shape():
    from litellm.types.utils import CallTypes

    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], enabled_models=["*-max"]
    )
    request = {
        "model": "github_copilot/claude-opus-5",
        "custom_llm_provider": "github_copilot",
        "tools": [{"type": "web_search"}],
        "litellm_metadata": {"deployment_model_name": "claude-opus-5-max"},
    }

    for call_type in (CallTypes.aresponses, CallTypes.responses):
        responses_result = await handler.async_pre_call_deployment_hook(
            dict(request), call_type
        )
        assert responses_result is not None
        assert responses_result["tools"][0]["name"] == "litellm_web_search"
        assert "function" not in responses_result["tools"][0]

    chat_result = await handler.async_pre_call_deployment_hook(
        dict(request), CallTypes.acompletion
    )
    assert chat_result is not None
    assert chat_result["tools"][0]["function"]["name"] == "litellm_web_search"


@pytest.mark.asyncio
async def test_direct_sdk_falls_back_to_requested_model_name():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["custom_provider"], enabled_models=["custom/claude-*"]
    )
    handler._execute_search = AsyncMock(return_value=("result", None))  # type: ignore[method-assign]

    result = await handler.try_short_circuit_search(
        model="custom/claude-opus-5",
        messages=[{"role": "user", "content": "query"}],
        tools=[{"type": "web_search"}],
        custom_llm_provider="custom_provider",
    )

    assert result is not None
