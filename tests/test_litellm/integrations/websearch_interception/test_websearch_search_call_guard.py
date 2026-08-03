"""
Test that a search LiteLLM runs on its own behalf is not intercepted.

When the search backend is a model, the search call carries the same
``web_search`` tool the deployment hook rewrites. Rewriting it would make the
search issue another search.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm._internal_context import is_internal_call, is_web_search_call
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)

WEB_SEARCH_REQUEST = {
    "model": "gpt-5.6-luna",
    "custom_llm_provider": "github_copilot",
    "tools": [{"type": "web_search"}],
}

REWRITTEN_TOOL_RESPONSE = {
    "content": [
        {
            "type": "tool_use",
            "name": "litellm_web_search",
            "input": {"query": "q"},
            "id": "toolu_1",
        }
    ]
}


@pytest.fixture
def handler():
    return WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], search_tool_name="llm-search"
    )


@pytest.mark.asyncio
async def test_client_request_is_still_rewritten(handler):
    result = await handler.async_pre_call_deployment_hook(
        dict(WEB_SEARCH_REQUEST), "aresponses"
    )

    assert result is not None
    assert result["tools"][0]["name"] == "litellm_web_search"


@pytest.mark.asyncio
async def test_search_call_tool_is_left_alone(handler):
    token = is_web_search_call.set(True)
    try:
        result = await handler.async_pre_call_deployment_hook(
            dict(WEB_SEARCH_REQUEST), "aresponses"
        )
    finally:
        is_web_search_call.reset(token)

    assert result is None


@pytest.mark.asyncio
async def test_messages_pre_request_hook_leaves_search_call_alone(handler):
    token = is_web_search_call.set(True)
    try:
        result = await handler.async_pre_request_hook(
            model="claude-opus-5",
            messages=[{"role": "user", "content": "q"}],
            kwargs={
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "litellm_params": {"custom_llm_provider": "github_copilot"},
            },
        )
    finally:
        is_web_search_call.reset(token)

    assert result is None


@pytest.mark.asyncio
async def test_interception_resumes_after_the_search_call(handler):
    token = is_web_search_call.set(True)
    is_web_search_call.reset(token)

    result = await handler.async_pre_call_deployment_hook(
        dict(WEB_SEARCH_REQUEST), "aresponses"
    )

    assert result is not None


@pytest.mark.asyncio
async def test_agentic_loop_does_not_run_for_a_search_call(handler):
    """
    The response-side gates have to be closed too: a search whose answer
    happens to carry a rewritten tool call must not start another loop.
    """
    args = dict(
        response=REWRITTEN_TOOL_RESPONSE,
        model="gpt-5.6-luna",
        messages=[],
        tools=[{"type": "web_search"}],
        stream=False,
        custom_llm_provider="github_copilot",
        kwargs={},
    )

    should_run, _ = await handler.async_should_run_agentic_loop(**args)
    assert should_run is True

    token = is_web_search_call.set(True)
    try:
        should_run, _ = await handler.async_should_run_agentic_loop(**args)
    finally:
        is_web_search_call.reset(token)
    assert should_run is False


@pytest.mark.asyncio
async def test_short_circuit_does_not_answer_a_search_call(handler):
    token = is_web_search_call.set(True)
    try:
        result = await handler.try_short_circuit_search(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "q"}],
            tools=[{"type": "web_search"}],
            custom_llm_provider="github_copilot",
        )
    finally:
        is_web_search_call.reset(token)

    assert result is None


def test_search_calls_are_still_billed():
    """
    The search sub-call is a real, paid model call. Marking it must not reuse
    ``is_internal_call``, which suppresses success logging and billing.
    """
    token = is_web_search_call.set(True)
    try:
        assert is_internal_call.get() is False
    finally:
        is_web_search_call.reset(token)
