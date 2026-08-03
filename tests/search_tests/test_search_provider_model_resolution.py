"""
Test that `search_provider` accepts a router model name in addition to the
`SearchProviders` enum members.

Enum members must keep resolving to their REST config even when a deployment
of the same name exists, and a name that matches neither must keep raising.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from litellm.search.main import _resolve_search_provider_as_model


@pytest.fixture
def router_with_llm_search_models(monkeypatch):
    import types

    router = Router(
        model_list=[
            {
                "model_name": "gpt-5.6-luna",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
            },
            {
                "model_name": "perplexity",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
            },
            {
                "model_name": "gpt-*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-fake"},
            },
        ],
        model_group_alias={"luna-alias": "gpt-5.6-luna"},
    )

    proxy_server = types.ModuleType("litellm.proxy.proxy_server")
    proxy_server.llm_router = router
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_server)
    return router


def test_resolves_model_group_name(router_with_llm_search_models):
    assert _resolve_search_provider_as_model("gpt-5.6-luna") == "gpt-5.6-luna"


def test_resolves_model_group_alias(router_with_llm_search_models):
    assert _resolve_search_provider_as_model("luna-alias") == "luna-alias"


def test_unknown_name_does_not_resolve(router_with_llm_search_models):
    assert _resolve_search_provider_as_model("no-such-model") is None


def test_wildcard_route_does_not_resolve(router_with_llm_search_models):
    """
    A mistyped provider must keep raising instead of being absorbed by a
    ``gpt-*`` deployment, which would send the search to a model that was
    never configured.
    """
    assert _resolve_search_provider_as_model("gpt-5.6-lun") is None


def test_resolution_without_router_returns_none():
    assert _resolve_search_provider_as_model("gpt-5.6-luna") is None


@pytest.mark.asyncio
async def test_enum_provider_wins_over_same_named_deployment(
    router_with_llm_search_models,
):
    """
    ``perplexity`` is both an enum member and a deployment in this router.
    The search meaning has to win, otherwise existing configs would silently
    start routing to an LLM.
    """
    import litellm

    with pytest.raises(Exception) as exc_info:
        await litellm.asearch(query="test", search_provider="perplexity")

    assert "resolves to model" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_unresolvable_provider_still_raises(router_with_llm_search_models):
    import litellm

    with pytest.raises(Exception, match="Search is not supported for provider"):
        await litellm.asearch(query="test", search_provider="gpt-5.6-lun")
