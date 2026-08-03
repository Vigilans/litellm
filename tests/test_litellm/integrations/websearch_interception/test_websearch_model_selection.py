"""
Test which models web search interception executes searches for.

Interception used to be all-or-nothing per provider. A provider hosts both
models that can search and models that cannot, so the decision has to be able
to name a model.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)

SEARCHING_MODEL = "gpt-5.6-luna"
NON_SEARCHING_MODEL = "claude-opus-5"


def test_provider_outside_the_enabled_list_is_never_intercepted():
    handler = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    assert not handler._should_intercept_model(
        model=NON_SEARCHING_MODEL, custom_llm_provider="github_copilot"
    )


def test_every_model_of_an_enabled_provider_is_intercepted_by_default():
    """
    The default has to stay all-or-nothing per provider, or enabling the new
    behaviour would change existing deployments.
    """
    handler = WebSearchInterceptionLogger(enabled_providers=["github_copilot"])

    assert handler._should_intercept_model(
        model=SEARCHING_MODEL, custom_llm_provider="github_copilot"
    )
    assert handler._should_intercept_model(
        model=NON_SEARCHING_MODEL, custom_llm_provider="github_copilot"
    )


def test_models_that_can_search_are_left_alone_when_asked():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], skip_models_with_web_search=True
    )

    assert not handler._should_intercept_model(
        model=SEARCHING_MODEL, custom_llm_provider="github_copilot"
    )
    assert handler._should_intercept_model(
        model=NON_SEARCHING_MODEL, custom_llm_provider="github_copilot"
    )


def test_a_deployment_can_force_interception_of_a_searching_model():
    """
    Outsourcing search from a slow or expensive model is the point of the
    per-deployment override, so it has to beat the capability check. The router
    spreads a deployment's litellm_params over the top level of the kwargs.
    """
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], skip_models_with_web_search=True
    )

    assert handler._should_intercept_model(
        model=SEARCHING_MODEL,
        custom_llm_provider="github_copilot",
        kwargs={"force_websearch_interception": True},
    )


def test_forcing_does_not_reach_across_providers():
    handler = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    assert not handler._should_intercept_model(
        model=SEARCHING_MODEL,
        custom_llm_provider="github_copilot",
        kwargs={"force_websearch_interception": True},
    )


@pytest.mark.asyncio
async def test_deployment_hook_leaves_a_searching_model_alone():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], skip_models_with_web_search=True
    )

    result = await handler.async_pre_call_deployment_hook(
        {
            "model": SEARCHING_MODEL,
            "custom_llm_provider": "github_copilot",
            "tools": [{"type": "web_search"}],
        },
        "aresponses",
    )

    assert result is None


@pytest.mark.asyncio
async def test_deployment_hook_rewrites_for_a_model_that_cannot_search():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], skip_models_with_web_search=True
    )

    result = await handler.async_pre_call_deployment_hook(
        {
            "model": NON_SEARCHING_MODEL,
            "custom_llm_provider": "github_copilot",
            "tools": [{"type": "web_search"}],
        },
        "acompletion",
    )

    assert result is not None
    assert result["tools"][0]["function"]["name"] == "litellm_web_search"


@pytest.mark.asyncio
async def test_deployment_hook_honours_the_per_deployment_override():
    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], skip_models_with_web_search=True
    )

    result = await handler.async_pre_call_deployment_hook(
        {
            "model": SEARCHING_MODEL,
            "custom_llm_provider": "github_copilot",
            "tools": [{"type": "web_search"}],
            "force_websearch_interception": True,
        },
        "aresponses",
    )

    assert result is not None
    assert result["tools"][0]["function"]["name"] == "litellm_web_search"


@pytest.mark.asyncio
async def test_router_deployment_override_reaches_the_hook(monkeypatch):
    """
    The router spreads litellm_params over the top level rather than nesting
    them, so a gate reading kwargs["litellm_params"] never sees the flag.
    """
    import litellm
    from litellm import Router

    handler = WebSearchInterceptionLogger(
        enabled_providers=["openai"], skip_models_with_web_search=True
    )
    monkeypatch.setattr(litellm, "callbacks", [handler])

    router = Router(
        model_list=[
            {
                "model_name": "probe",
                "litellm_params": {
                    "model": f"openai/{SEARCHING_MODEL}",
                    "api_key": "sk-fake",
                    "force_websearch_interception": True,
                    "mock_response": "ok",
                },
            }
        ]
    )

    intercepted = {}
    original = handler.async_pre_call_deployment_hook

    async def record(kwargs, call_type):
        result = await original(kwargs, call_type)
        intercepted["rewritten"] = result is not None
        return result

    monkeypatch.setattr(handler, "async_pre_call_deployment_hook", record)

    await router.acompletion(
        model="probe",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "web_search"}],
    )

    assert intercepted["rewritten"] is True


def test_config_yaml_carries_the_new_setting():
    handler = WebSearchInterceptionLogger.from_config_yaml(
        {
            "enabled_providers": ["github_copilot"],
            "search_tool_name": "llm-search",
            "skip_models_with_web_search": True,
        }
    )

    assert handler.skip_models_with_web_search is True
    assert not handler._should_intercept_model(
        model=SEARCHING_MODEL, custom_llm_provider="github_copilot"
    )


@pytest.mark.parametrize(
    "model, supports",
    [
        (SEARCHING_MODEL, True),
        (f"github_copilot/{SEARCHING_MODEL}", True),
        (NON_SEARCHING_MODEL, False),
        ("github_copilot/claude-opus-4.5", False),
        ("model-that-is-not-registered", False),
    ],
)
def test_capability_lookup_reads_the_registry_directly(model, supports, monkeypatch):
    """
    Resolving a provider authenticates it, and GitHub Copilot's authentication
    blocks on an interactive device-code prompt when no token is cached. These
    gates run on every request, so the lookup must not go through provider
    resolution.
    """
    import litellm

    def fail(*args, **kwargs):
        raise AssertionError("provider resolution must not run in the gate")

    monkeypatch.setattr(litellm, "get_llm_provider", fail)
    monkeypatch.setattr(litellm, "supports_web_search", fail)

    handler = WebSearchInterceptionLogger(
        enabled_providers=["github_copilot"], skip_models_with_web_search=True
    )

    assert (
        handler._model_supports_web_search(
            model=model, custom_llm_provider="github_copilot"
        )
        is supports
    )


def test_capability_lookup_matches_supports_web_search():
    """
    The registry lookup stands in for ``litellm.supports_web_search``, so it
    has to agree with it; xAI, for one, declares the capability at provider
    level rather than on each model entry.
    """
    import litellm

    handler = WebSearchInterceptionLogger(enabled_providers=["xai"])
    providers = {
        "openai": "openai",
        "anthropic": "anthropic",
        "bedrock": "bedrock",
        "vertex_ai-language-models": "vertex_ai",
        "gemini": "gemini",
        "xai": "xai",
    }

    mismatches = []
    for name, entry in list(litellm.model_cost.items()):
        if not isinstance(entry, dict):
            continue
        provider = providers.get(entry.get("litellm_provider"))
        if provider is None:
            continue
        expected = litellm.supports_web_search(name, provider)
        if handler._model_supports_web_search(name, provider) != expected:
            mismatches.append(name)

    assert mismatches == []
