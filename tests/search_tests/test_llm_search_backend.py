import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.search._context import search_router
from litellm.search.model_adapter.handler import (
    SearchModelEndpoint,
    _call_selected_search_model,
    execute_model_search,
    select_search_model_endpoint,
)
from litellm.search.model_adapter.transformation import (
    chat_to_search_response,
    messages_to_search_response,
    responses_to_search_response,
)

ANSWER = (
    "First claim.([a](https://example.com/a))"
    "Second claim.([b](https://example.com/b))"
)
FIRST_MARKER_START = ANSWER.index("([a]")
SECOND_MARKER_START = ANSWER.index("([b]")
ANNOTATIONS = [
    {
        "type": "url_citation",
        "title": "A",
        "url": "https://example.com/a",
        "start_index": FIRST_MARKER_START,
        "end_index": ANSWER.index("Second claim."),
    },
    {
        "type": "url_citation",
        "title": "B",
        "url": "https://example.com/b",
        "start_index": SECOND_MARKER_START,
        "end_index": len(ANSWER),
    },
]


def test_responses_annotations_use_cited_prose():
    response = SimpleNamespace(
        output=[
            {
                "type": "message",
                "content": [
                    {"text": ANSWER, "annotations": list(reversed(ANNOTATIONS))}
                ],
            }
        ]
    )

    result = responses_to_search_response(response, query="q")

    assert [item.url for item in result.results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [item.snippet for item in result.results] == [
        "First claim.",
        "Second claim.",
    ]
    assert result._hidden_params["response_cost"] == 0.0


def test_responses_preserves_each_citation_for_the_same_url():
    text = "First claim.[a]Second claim.[a]"
    response = SimpleNamespace(
        output=[
            {
                "type": "message",
                "content": [
                    {
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "A",
                                "url": "https://example.com/a",
                                "start_index": 12,
                                "end_index": 15,
                            },
                            {
                                "type": "url_citation",
                                "title": "A",
                                "url": "https://example.com/a",
                                "start_index": 28,
                                "end_index": 31,
                            },
                        ],
                    }
                ],
            }
        ]
    )

    result = responses_to_search_response(response, query="q")

    assert [item.url for item in result.results] == [
        "https://example.com/a",
        "https://example.com/a",
    ]
    assert [item.snippet for item in result.results] == [
        "First claim.",
        "Second claim.",
    ]


def test_responses_missing_indices_do_not_swallow_later_snippet():
    response = SimpleNamespace(
        output=[
            {
                "type": "message",
                "content": [
                    {
                        "text": ANSWER,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "No offsets",
                                "url": "https://example.com/no-offsets",
                            },
                            ANNOTATIONS[1],
                        ],
                    }
                ],
            }
        ]
    )

    result = responses_to_search_response(response, query="q")

    assert result.results[0].snippet == ""
    assert result.results[1].snippet == ANSWER[:SECOND_MARKER_START]


def test_responses_collects_multiple_content_parts():
    response = SimpleNamespace(
        output=[
            {
                "type": "message",
                "content": [
                    {
                        "text": "First.[a]",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "A",
                                "url": "https://example.com/a",
                                "start_index": 6,
                                "end_index": 9,
                            }
                        ],
                    },
                    {
                        "text": "Second.[b]",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "B",
                                "url": "https://example.com/b",
                                "start_index": 7,
                                "end_index": 10,
                            }
                        ],
                    },
                ],
            }
        ]
    )

    result = responses_to_search_response(response, query="q")

    assert [item.snippet for item in result.results] == ["First.", "Second."]


def test_chat_combines_annotations_and_structured_results():
    nested_citation = SimpleNamespace(
        title="A",
        url="https://example.com/a",
        start_index=6,
        end_index=9,
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Claim.[a]",
                    annotations=[
                        SimpleNamespace(
                            type="url_citation", url_citation=nested_citation
                        )
                    ],
                    provider_specific_fields={
                        "web_search_results": [
                            {
                                "type": "web_search_tool_result",
                                "content": [
                                    {
                                        "type": "web_search_result",
                                        "title": "A duplicate",
                                        "url": "https://example.com/a",
                                        "page_age": "2026-08-01",
                                    },
                                    {
                                        "type": "web_search_result",
                                        "title": "B",
                                        "url": "https://example.com/b",
                                        "snippet": "Second source",
                                    },
                                ],
                            }
                        ]
                    },
                )
            )
        ]
    )

    result = chat_to_search_response(response, query="q")

    assert [item.url for item in result.results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert result.results[0].title == "A"
    assert result.results[0].snippet == "Claim."
    assert result.results[0].date == "2026-08-01"
    assert result.results[1].snippet == "Second source"


def test_chat_flattens_pydantic_provider_citations():
    citation = SimpleNamespace(
        type="web_search_result_location",
        title="A",
        url="https://example.com/a",
        cited_text="First source",
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Answer",
                    annotations=[],
                    provider_specific_fields={"citations": [[citation]]},
                )
            )
        ]
    )

    result = chat_to_search_response(response, query="q")

    assert result.results[0].url == "https://example.com/a"
    assert result.results[0].snippet == "First source"


def test_messages_use_web_search_citations_and_result_blocks():
    response = {
        "content": [
            {
                "type": "text",
                "text": "Answer",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "title": "A",
                        "url": "https://example.com/a",
                        "cited_text": "First source",
                    },
                    {
                        "type": "page_location",
                        "document_title": "Not a web result",
                    },
                ],
            },
            {
                "type": "web_search_tool_result",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "B",
                        "url": "https://example.com/b",
                        "page_age": "2026-08-02",
                    }
                ],
            },
        ]
    }

    result = messages_to_search_response(response, query="q")

    assert [item.url for item in result.results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert result.results[0].snippet == "First source"
    assert result.results[1].date == "2026-08-02"


@pytest.mark.parametrize(
    "model,provider,model_info,expected",
    [
        (
            "custom-name",
            "openai_like",
            {"mode": "responses"},
            SearchModelEndpoint.RESPONSES,
        ),
        (
            "gpt-5.6-luna",
            "github_copilot",
            {},
            SearchModelEndpoint.RESPONSES,
        ),
        (
            "claude-opus-5",
            "github_copilot",
            {},
            SearchModelEndpoint.ANTHROPIC_MESSAGES,
        ),
        (
            "other-model",
            "openai_like",
            {},
            SearchModelEndpoint.CHAT,
        ),
    ],
)
def test_selects_search_endpoint(model, provider, model_info, expected):
    assert (
        select_search_model_endpoint(
            model=model,
            custom_llm_provider=provider,
            model_info=model_info,
        )
        == expected
    )


def test_unknown_provider_uses_chat_without_authentication():
    assert (
        select_search_model_endpoint(
            model="custom-model",
            custom_llm_provider="custom-provider",
            model_info={},
        )
        == SearchModelEndpoint.CHAT
    )


@pytest.mark.asyncio
async def test_model_search_internal_params_are_not_logged(monkeypatch):
    import importlib

    from litellm.llms.base_llm.search.transformation import SearchResponse

    search_main = importlib.import_module("litellm.search.main")

    class LoggingObj:
        def __init__(self):
            self.optional_params = None

        def update_from_kwargs(self, *, optional_params, **kwargs):
            self.optional_params = optional_params

    logging_obj = LoggingObj()

    async def fake_execute_model_search(**kwargs):
        assert kwargs["num_retries"] == 3
        return SearchResponse(results=[])

    monkeypatch.setattr(search_main, "execute_model_search", fake_execute_model_search)

    token = search_router.set(
        SimpleNamespace(get_model_names=lambda: ["llm-search-model"])
    )
    try:
        response = search_main.search.__wrapped__(
            query="q",
            search_provider="llm-search-model",
            asearch=True,
            litellm_logging_obj=logging_obj,
            _model_search_num_retries=3,
        )
    finally:
        search_router.reset(token)

    assert await response == SearchResponse(results=[])
    assert "_model_search_num_retries" not in logging_obj.optional_params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,expected_call",
    [
        (SearchModelEndpoint.RESPONSES, "aresponses"),
        (SearchModelEndpoint.ANTHROPIC_MESSAGES, "anthropic_messages"),
        (SearchModelEndpoint.CHAT, "acompletion"),
    ],
)
async def test_selected_endpoint_request_is_native_and_isolated(
    monkeypatch, endpoint, expected_call
):
    import importlib

    import litellm
    from litellm._internal_context import is_web_search_call

    handler_module = importlib.import_module("litellm.search.model_adapter.handler")
    monkeypatch.setattr(
        handler_module,
        "select_search_model_endpoint",
        lambda **kwargs: endpoint,
    )
    seen = {}

    async def fake_endpoint(**kwargs):
        seen.update(kwargs)
        seen["marked"] = is_web_search_call.get()
        if endpoint == SearchModelEndpoint.RESPONSES:
            return SimpleNamespace(output=[])
        if endpoint == SearchModelEndpoint.ANTHROPIC_MESSAGES:
            return {"content": []}
        return SimpleNamespace(choices=[])

    for name in ("aresponses", "anthropic_messages", "acompletion"):
        monkeypatch.setattr(
            litellm,
            name,
            fake_endpoint if name == expected_call else AsyncMock(),
        )

    outer_metadata = {"user_api_key_team_id": "team-1"}
    result = await _call_selected_search_model(
        query="q",
        model="deployment-model",
        custom_llm_provider="github_copilot",
        api_key="test-key",
        api_version="2026-08-01",
        max_tokens=2048,
        tools=[{"type": "function", "name": "deployment-tool"}],
        input="deployment input",
        messages=[{"role": "user", "content": "deployment message"}],
        web_search_options={"search_context_size": "low"},
        model_info={"id": "deployment-1"},
        metadata={
            "user_api_key_user_id": "user-1",
            "user_api_key_budget_reservation": {"finalized": False},
        },
        litellm_metadata=outer_metadata,
        litellm_trace_id="trace-1",
        litellm_session_id="session-1",
        litellm_logging_obj=object(),
        litellm_call_id="outer-call",
    )

    assert seen["model"] == "deployment-model"
    assert seen["stream"] is False
    assert seen["marked"] is True
    expected_metadata = {
        "user_api_key_user_id": "user-1",
        "user_api_key_team_id": "team-1",
    }
    assert seen["metadata"] == expected_metadata
    assert seen["litellm_metadata"] == expected_metadata
    assert seen["litellm_metadata"] is not outer_metadata
    assert "user_api_key_budget_reservation" not in seen["litellm_metadata"]
    assert seen["litellm_trace_id"] == "trace-1"
    assert seen["litellm_session_id"] == "session-1"
    assert seen["api_version"] == "2026-08-01"
    assert "litellm_logging_obj" not in seen
    assert "litellm_call_id" not in seen
    assert result.results == []
    assert is_web_search_call.get() is False

    if endpoint == SearchModelEndpoint.RESPONSES:
        expected_request = {
            "model": "deployment-model",
            "stream": False,
            "input": "q",
            "tools": [{"type": "web_search"}],
        }
    elif endpoint == SearchModelEndpoint.ANTHROPIC_MESSAGES:
        expected_request = {
            "model": "deployment-model",
            "stream": False,
            "messages": [{"role": "user", "content": "q"}],
            "max_tokens": 2048,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
    else:
        expected_request = {
            "model": "deployment-model",
            "stream": False,
            "messages": [{"role": "user", "content": "q"}],
            "web_search_options": {},
        }

    for key, value in expected_request.items():
        assert seen[key] == value
    assert seen["proxy_server_request"] == {"body": expected_request}


@pytest.mark.asyncio
async def test_search_guard_resets_when_endpoint_fails(monkeypatch):
    import importlib

    import litellm
    from litellm._internal_context import is_web_search_call

    handler_module = importlib.import_module("litellm.search.model_adapter.handler")
    monkeypatch.setattr(
        handler_module,
        "select_search_model_endpoint",
        lambda **kwargs: SearchModelEndpoint.CHAT,
    )

    async def fail(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "acompletion", fail)

    with pytest.raises(RuntimeError, match="provider down"):
        await _call_selected_search_model(
            query="q", model="m", custom_llm_provider="openai_like"
        )
    assert is_web_search_call.get() is False


@pytest.mark.asyncio
async def test_model_search_uses_router_retry_without_model_fallback():
    class FakeRouter:
        num_retries = 4

        def __init__(self):
            self.seen = None

        async def _ageneric_api_call_with_fallbacks(self, **kwargs):
            self.seen = kwargs
            return messages_to_search_response(
                {"content": [{"type": "text", "text": "answer"}]}, query="q"
            )

    router = FakeRouter()
    outer_metadata = {
        "user_api_key_team_id": "team-1",
        "nested": {"value": 1},
    }
    token = search_router.set(router)
    try:
        result = await execute_model_search(
            query="q",
            model="llm-search-model",
            timeout=12,
            num_retries=3,
            request_kwargs={
                "litellm_metadata": outer_metadata,
                "litellm_logging_obj": object(),
                "litellm_call_id": "outer-call",
            },
        )
    finally:
        search_router.reset(token)

    assert router.seen["model"] == "llm-search-model"
    assert router.seen["timeout"] == 12
    assert router.seen["num_retries"] == 3
    assert router.seen["disable_fallbacks"] is True
    assert router.seen["litellm_metadata"] == {
        "user_api_key_team_id": "team-1",
        "nested": {"value": 1},
    }
    router.seen["litellm_metadata"]["nested"]["value"] = 2
    assert outer_metadata["nested"]["value"] == 1
    assert "litellm_logging_obj" not in router.seen
    assert "litellm_call_id" not in router.seen
    assert result.results[0].snippet == "answer"


@pytest.mark.asyncio
async def test_model_search_requires_router():
    with pytest.raises(ValueError, match="requires a Router"):
        await execute_model_search(query="q", model="llm-search-model")


def test_uncited_answer_is_preserved():
    result = chat_to_search_response(
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="The answer.",
                        annotations=[],
                        provider_specific_fields={},
                    )
                )
            ]
        ),
        query="q",
    )

    assert [(item.title, item.url, item.snippet) for item in result.results] == [
        ("q", "", "The answer.")
    ]
