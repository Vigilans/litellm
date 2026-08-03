import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from litellm.search._context import search_router
from litellm.search.main import _resolve_search_provider_as_model


@pytest.fixture
def router_with_llm_search_models():
    return Router(
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
        search_tools=[
            {
                "search_tool_name": "llm-search",
                "litellm_params": {"search_provider": "gpt-5.6-luna"},
            },
            {
                "search_tool_name": "perplexity",
                "litellm_params": {"search_provider": "perplexity"},
            },
            {
                "search_tool_name": "bad-search",
                "litellm_params": {"search_provider": "gpt-5.6-lun"},
            },
        ],
    )


def _resolve_with_router(router: Router, search_provider: str):
    token = search_router.set(router)
    try:
        return _resolve_search_provider_as_model(search_provider)
    finally:
        search_router.reset(token)


def test_resolves_model_group_name(router_with_llm_search_models):
    assert (
        _resolve_with_router(router_with_llm_search_models, "gpt-5.6-luna")
        == "gpt-5.6-luna"
    )


def test_resolves_model_group_alias(router_with_llm_search_models):
    assert (
        _resolve_with_router(router_with_llm_search_models, "luna-alias")
        == "luna-alias"
    )


def test_unknown_name_does_not_resolve(router_with_llm_search_models):
    assert _resolve_with_router(router_with_llm_search_models, "no-such-model") is None


def test_wildcard_route_does_not_resolve(router_with_llm_search_models):
    assert _resolve_with_router(router_with_llm_search_models, "gpt-5.6-lun") is None


def test_resolution_without_router_returns_none():
    assert _resolve_search_provider_as_model("gpt-5.6-luna") is None


@pytest.mark.asyncio
async def test_enum_provider_wins_over_same_named_deployment(
    router_with_llm_search_models, monkeypatch
):
    import importlib

    search_main = importlib.import_module("litellm.search.main")
    called = False

    def fake_get_provider_search_config(*, provider):
        nonlocal called
        called = True
        raise RuntimeError("rest provider selected")

    monkeypatch.setattr(
        search_main.ProviderConfigManager,
        "get_provider_search_config",
        fake_get_provider_search_config,
    )

    with pytest.raises(Exception, match="rest provider selected"):
        await router_with_llm_search_models.asearch(
            query="test", search_tool_name="perplexity", num_retries=0
        )

    assert called is True


@pytest.mark.asyncio
async def test_router_supplies_model_resolution_context(
    router_with_llm_search_models, monkeypatch
):
    from litellm.llms.base_llm.search.transformation import SearchResponse

    async def fake_asearch(*, search_provider, **kwargs):
        assert search_router.get() is router_with_llm_search_models
        assert _resolve_search_provider_as_model(search_provider) == search_provider
        return SearchResponse(results=[])

    monkeypatch.setattr(router_with_llm_search_models, "num_retries", 0)
    router_with_llm_search_models.asearch = (
        router_with_llm_search_models.factory_function(
            fake_asearch, call_type="asearch"
        )
    )

    response = await router_with_llm_search_models.asearch(
        query="test", search_tool_name="llm-search"
    )

    assert response.results == []
    assert search_router.get() is None


@pytest.mark.asyncio
async def test_unresolvable_provider_still_raises(router_with_llm_search_models):
    with pytest.raises(Exception, match="Search is not supported for provider"):
        await router_with_llm_search_models.asearch(
            query="test", search_tool_name="bad-search", num_retries=0
        )
