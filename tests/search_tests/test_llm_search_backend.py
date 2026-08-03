"""
Test the LLM search backend: turning a model's cited answer into a
``SearchResponse``.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.search.llm_search import (
    _annotations_to_results,
    _extract_text_and_annotations,
    execute_llm_search,
)

ANSWER = (
    "Here are three headlines:\n\n"
    "1. **Anthropic says its AI models hacked three organizations.**  \n"
    "   The incidents involved Claude models that accessed real systems."
    "([apnews.com](https://apnews.com/article/b0a2?utm_source=openai))"
    "2. **The EU begins enforcing AI Act provisions on deepfakes.**  \n"
    "   New requirements include clearer labeling for AI-generated material."
    "([digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/ai))"
)

FIRST_MARKER_START = ANSWER.index("([apnews.com]")
FIRST_MARKER_END = ANSWER.index("2. **The EU")
SECOND_MARKER_START = ANSWER.index("([digital-strategy")

ANNOTATIONS = [
    {
        "type": "url_citation",
        "title": "Anthropic says its AI models hacked 3 organizations",
        "url": "https://apnews.com/article/b0a2?utm_source=openai",
        "start_index": FIRST_MARKER_START,
        "end_index": FIRST_MARKER_END,
    },
    {
        "type": "url_citation",
        "title": "AI Act | Shaping Europe's digital future",
        "url": "https://digital-strategy.ec.europa.eu/ai",
        "start_index": SECOND_MARKER_START,
        "end_index": len(ANSWER),
    },
]


def test_snippet_is_the_cited_prose_not_the_marker():
    """
    ``start_index``/``end_index`` cover the citation marker itself, so slicing
    that span yields a bare markdown link. The snippet has to come from the
    text leading up to the marker.
    """
    results = _annotations_to_results(text=ANSWER, annotations=ANNOTATIONS)

    assert len(results) == 2
    assert "Anthropic says its AI models hacked three organizations" in (
        results[0].snippet
    )
    assert "apnews.com" not in results[0].snippet
    assert "EU begins enforcing AI Act provisions" in results[1].snippet
    assert "digital-strategy" not in results[1].snippet


def test_title_and_url_come_from_the_annotation():
    results = _annotations_to_results(text=ANSWER, annotations=ANNOTATIONS)

    assert results[0].url == "https://apnews.com/article/b0a2?utm_source=openai"
    assert results[0].title.startswith("Anthropic says")


def test_repeated_url_is_reported_once():
    duplicated = ANNOTATIONS + [dict(ANNOTATIONS[0])]

    results = _annotations_to_results(text=ANSWER, annotations=duplicated)

    assert [r.url for r in results] == [
        "https://apnews.com/article/b0a2?utm_source=openai",
        "https://digital-strategy.ec.europa.eu/ai",
    ]


def test_non_citation_annotations_are_skipped():
    results = _annotations_to_results(
        text=ANSWER,
        annotations=[{"type": "file_citation", "file_id": "f-1"}] + ANNOTATIONS,
    )

    assert len(results) == 2


def test_no_annotations_yields_no_results():
    assert _annotations_to_results(text=ANSWER, annotations=[]) == []


def test_out_of_order_annotations_get_their_own_prose():
    """
    A model may report citations in any order. Walking them unsorted moves the
    snippet cursor backwards and hands one source another's text.
    """
    results = _annotations_to_results(
        text=ANSWER, annotations=list(reversed(ANNOTATIONS))
    )

    by_url = {r.url: r.snippet for r in results}
    assert "Anthropic says its AI models hacked three organizations" in (
        by_url["https://apnews.com/article/b0a2?utm_source=openai"]
    )
    assert "EU begins enforcing AI Act provisions" in (
        by_url["https://digital-strategy.ec.europa.eu/ai"]
    )


def test_missing_indices_do_not_swallow_the_next_snippet():
    annotations = [
        {
            "type": "url_citation",
            "title": "No offsets",
            "url": "https://example.com/a",
        },
        ANNOTATIONS[1],
    ]

    results = _annotations_to_results(text=ANSWER, annotations=annotations)

    assert results[0].snippet == ""
    assert "EU begins enforcing AI Act provisions" in results[1].snippet


def test_extracts_across_multiple_message_parts():
    """
    ``output`` items may be plain dicts, and a message may arrive in several
    content parts; annotation offsets are relative to their own part.
    """
    response = SimpleNamespace(
        output=[
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [
                    {
                        "text": "First claim.[a]",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "A",
                                "url": "https://example.com/a",
                                "start_index": len("First claim."),
                                "end_index": len("First claim.[a]"),
                            }
                        ],
                    },
                    {
                        "text": "Second claim.[b]",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "B",
                                "url": "https://example.com/b",
                                "start_index": len("Second claim."),
                                "end_index": len("Second claim.[b]"),
                            }
                        ],
                    },
                ],
            },
        ]
    )

    text, annotations = _extract_text_and_annotations(response)
    results = _annotations_to_results(text=text, annotations=annotations)

    assert [r.url for r in results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert results[0].snippet == "First claim."
    assert results[1].snippet == "Second claim."


def test_response_without_a_message_item_extracts_nothing():
    response = SimpleNamespace(output=[{"type": "web_search_call", "status": "done"}])

    assert _extract_text_and_annotations(response) == ("", [])


@pytest.mark.asyncio
async def test_search_call_is_marked_and_never_streams(monkeypatch):
    """
    The request has to be marked so web search interception leaves its own
    search alone, and it must not stream: a streaming response carries no
    ``output`` to read citations from.
    """
    import litellm
    from litellm._internal_context import is_web_search_call

    seen = {}

    async def fake_aresponses(**kwargs):
        seen["kwargs"] = kwargs
        seen["marked"] = is_web_search_call.get()
        return SimpleNamespace(output=[])

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    await execute_llm_search(query="q", model="gpt-5.6-luna")

    assert seen["marked"] is True
    assert seen["kwargs"]["stream"] is False
    assert seen["kwargs"]["tools"] == [{"type": "web_search"}]
    assert is_web_search_call.get() is False


@pytest.mark.asyncio
async def test_mark_is_cleared_when_the_model_call_fails(monkeypatch):
    import litellm
    from litellm._internal_context import is_web_search_call

    async def failing_aresponses(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "aresponses", failing_aresponses)

    with pytest.raises(RuntimeError):
        await execute_llm_search(query="q", model="gpt-5.6-luna")

    assert is_web_search_call.get() is False


@pytest.mark.asyncio
async def test_uncited_answer_is_still_returned(monkeypatch):
    """
    Some providers search but report no citations. Dropping the answer would
    leave the caller with nothing to feed back to the model.
    """
    import litellm

    async def uncited_aresponses(**kwargs):
        return SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "content": [{"text": "The answer.", "annotations": []}],
                }
            ]
        )

    monkeypatch.setattr(litellm, "aresponses", uncited_aresponses)

    response = await execute_llm_search(query="q", model="gpt-5.6-luna")

    assert len(response.results) == 1
    assert response.results[0].snippet == "The answer."
