from typing import Any, Dict, Iterable, List, Optional

from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        return dumped if isinstance(dumped, dict) else {}
    try:
        return dict(vars(value))
    except (TypeError, AttributeError):
        return {}


def _append_result(
    results: List[SearchResult],
    seen_urls: Dict[str, SearchResult],
    *,
    title: Optional[str],
    url: Optional[str],
    snippet: Optional[str],
    date: Optional[str] = None,
    last_updated: Optional[str] = None,
) -> None:
    if not url:
        return

    existing = seen_urls.get(url)
    if existing is not None:
        if not existing.snippet and snippet:
            existing.snippet = snippet
        if existing.date is None and date is not None:
            existing.date = date
        if existing.last_updated is None and last_updated is not None:
            existing.last_updated = last_updated
        return

    result = SearchResult(
        title=title or url,
        url=url,
        snippet=snippet or "",
        date=date,
        last_updated=last_updated,
    )
    results.append(result)
    seen_urls[url] = result


def _annotation_values(annotation: Any) -> Dict[str, Any]:
    values = _as_dict(annotation)
    nested = _as_dict(values.get("url_citation"))
    if nested:
        return {**values, **nested}
    return values


def _add_indexed_citations(
    *,
    text: str,
    annotations: Iterable[Any],
    results: List[SearchResult],
    seen_urls: Dict[str, SearchResult],
) -> None:
    citations: List[Dict[str, Any]] = []
    for annotation in annotations:
        values = _annotation_values(annotation)
        if values.get("url") and values.get("type", "url_citation") == "url_citation":
            citations.append(values)

    def _start_index(item: Dict[str, Any]) -> int:
        value = item.get("start_index")
        return value if isinstance(value, int) else 0

    citations.sort(key=_start_index)

    snippet_start = 0
    for citation in citations:
        marker_start = citation.get("start_index")
        if not isinstance(marker_start, int):
            marker_start = 0
        marker_start = max(0, marker_start)

        marker_end = citation.get("end_index")
        if not isinstance(marker_end, int):
            marker_end = marker_start
        marker_end = max(marker_start, marker_end)

        snippet = (
            citation.get("supported_text")
            or citation.get("cited_text")
            or text[snippet_start:marker_start].strip()
        )
        snippet_start = max(snippet_start, marker_end)
        url = citation.get("url")
        result = SearchResult(
            title=citation.get("title") or url,
            url=url,
            snippet=snippet,
        )
        results.append(result)
        seen_urls.setdefault(url, result)


def _iter_nested_values(values: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _iter_nested_values(value)
        return

    value = _as_dict(values)
    if not value:
        return

    yield value
    content = value.get("content")
    if isinstance(content, (dict, list, tuple)):
        yield from _iter_nested_values(content)


def _add_structured_results(
    *,
    values: Any,
    results: List[SearchResult],
    seen_urls: Dict[str, SearchResult],
) -> None:
    for value in _iter_nested_values(values):
        if not value.get("url"):
            continue
        _append_result(
            results,
            seen_urls,
            title=value.get("title"),
            url=value.get("url"),
            snippet=(
                value.get("snippet")
                or value.get("cited_text")
                or value.get("supported_text")
            ),
            date=value.get("page_age") or value.get("date"),
            last_updated=value.get("last_updated"),
        )


def _extract_chat_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: List[str] = []
    for part in content:
        part_dict = _as_dict(part)
        if part_dict.get("type") in ("text", "output_text"):
            text_parts.append(part_dict.get("text") or "")
    return "".join(text_parts)


def _finish_response(
    *, query: str, text: str, results: List[SearchResult]
) -> SearchResponse:
    if not results and text:
        results.append(SearchResult(title=query, url="", snippet=text))
    response = SearchResponse(results=results)
    response._hidden_params["response_cost"] = 0.0
    return response


def responses_to_search_response(response: Any, query: str) -> SearchResponse:
    response_dict = _as_dict(response)
    output = response_dict.get("output") or getattr(response, "output", None) or []
    results: List[SearchResult] = []
    seen_urls: Dict[str, SearchResult] = {}
    text_parts: List[str] = []

    for item in output:
        item_dict = _as_dict(item)
        if item_dict.get("type") != "message":
            continue
        for content in item_dict.get("content") or []:
            content_dict = _as_dict(content)
            text = content_dict.get("text") or ""
            text_parts.append(text)
            _add_indexed_citations(
                text=text,
                annotations=content_dict.get("annotations") or [],
                results=results,
                seen_urls=seen_urls,
            )

    return _finish_response(query=query, text="".join(text_parts), results=results)


def chat_to_search_response(response: Any, query: str) -> SearchResponse:
    response_dict = _as_dict(response)
    choices = response_dict.get("choices") or getattr(response, "choices", None) or []
    results: List[SearchResult] = []
    seen_urls: Dict[str, SearchResult] = {}
    text_parts: List[str] = []

    for choice in choices:
        choice_dict = _as_dict(choice)
        message = _as_dict(
            choice_dict.get("message") or getattr(choice, "message", None)
        )
        text = _extract_chat_text(message.get("content"))
        text_parts.append(text)
        _add_indexed_citations(
            text=text,
            annotations=message.get("annotations") or [],
            results=results,
            seen_urls=seen_urls,
        )

        provider_fields = _as_dict(message.get("provider_specific_fields"))
        _add_structured_results(
            values=provider_fields.get("web_search_results"),
            results=results,
            seen_urls=seen_urls,
        )
        _add_structured_results(
            values=provider_fields.get("citations"),
            results=results,
            seen_urls=seen_urls,
        )

    return _finish_response(query=query, text="".join(text_parts), results=results)


def messages_to_search_response(response: Any, query: str) -> SearchResponse:
    response_dict = _as_dict(response)
    results: List[SearchResult] = []
    seen_urls: Dict[str, SearchResult] = {}
    text_parts: List[str] = []

    for content in response_dict.get("content") or []:
        content_dict = _as_dict(content)
        content_type = content_dict.get("type")
        if content_type == "text":
            text_parts.append(content_dict.get("text") or "")
            _add_structured_results(
                values=content_dict.get("citations"),
                results=results,
                seen_urls=seen_urls,
            )
        elif content_type == "web_search_tool_result":
            _add_structured_results(
                values=content_dict.get("content"),
                results=results,
                seen_urls=seen_urls,
            )

    return _finish_response(query=query, text="".join(text_parts), results=results)
