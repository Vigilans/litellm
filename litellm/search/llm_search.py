"""
Search execution backed by an LLM that searches the web natively.
"""

from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult

WEB_SEARCH_TOOL: Dict[str, Any] = {"type": "web_search"}


def _annotations_to_results(text: str, annotations: List[Any]) -> List[SearchResult]:
    """
    Build search results from ``url_citation`` annotations.

    ``start_index``/``end_index`` span the citation marker itself, not the
    material being cited, so the snippet is taken from the answer text between
    the previous marker and this one. Markers are sorted by position first
    because a model may report them out of order.
    """
    citations: List[Dict[str, Any]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            annotation = annotation.model_dump()
        if annotation.get("type") == "url_citation" and annotation.get("url"):
            citations.append(annotation)

    citations.sort(key=lambda a: a.get("start_index") or 0)

    results: List[SearchResult] = []
    seen_urls: set = set()
    snippet_start = 0

    for citation in citations:
        marker_start = citation.get("start_index") or 0
        marker_end = citation.get("end_index") or marker_start
        snippet = text[snippet_start:marker_start].strip()
        snippet_start = max(snippet_start, marker_end)

        url = citation["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        results.append(
            SearchResult(
                title=citation.get("title") or url,
                url=url,
                snippet=snippet,
            )
        )

    return results


def _extract_text_and_annotations(response: Any) -> tuple:
    """
    Collect the answer text and its citations across every message part.

    ``ResponsesAPIResponse.output`` items are typed models or plain dicts
    depending on the provider path.
    """
    texts: List[str] = []
    annotations: List[Any] = []
    offset = 0

    for item in getattr(response, "output", None) or []:
        item_dict = item if isinstance(item, dict) else item.model_dump()
        if item_dict.get("type") != "message":
            continue
        for content in item_dict.get("content") or []:
            text = content.get("text") or ""
            for annotation in content.get("annotations") or []:
                if not isinstance(annotation, dict):
                    annotation = dict(annotation)
                if offset:
                    for key in ("start_index", "end_index"):
                        if annotation.get(key) is not None:
                            annotation[key] += offset
                annotations.append(annotation)
            texts.append(text)
            offset += len(text)

    return "".join(texts), annotations


async def execute_llm_search(
    query: str,
    model: str,
    timeout: Optional[Any] = None,
) -> SearchResponse:
    """
    Run a search by asking ``model`` to search the web and citing its sources.

    The request carries a hosted ``web_search`` tool because that is the only
    shape the providers act on; ``web_search_options`` either yields no
    citations or is ignored outright. Web search interception rewrites that
    same tool, so the call is marked internal to keep it from being
    intercepted and recursing back into this function.
    """
    import litellm
    from litellm._internal_context import is_web_search_call

    router = None
    try:
        from litellm.proxy.proxy_server import llm_router

        router = llm_router
    except ImportError:
        verbose_logger.debug(
            "LLM search: could not import llm_router, calling litellm.aresponses directly"
        )

    request: Dict[str, Any] = {
        "model": model,
        "input": query,
        "tools": [WEB_SEARCH_TOOL],
        "stream": False,
    }
    if timeout is not None:
        request["timeout"] = timeout

    token = is_web_search_call.set(True)
    try:
        if router is not None:
            response = await router.aresponses(**request)
        else:
            response = await litellm.aresponses(**request)
    finally:
        is_web_search_call.reset(token)

    text, annotations = _extract_text_and_annotations(response)
    results = _annotations_to_results(text=text, annotations=annotations)

    if not results and text:
        verbose_logger.debug(
            f"LLM search: '{model}' answered '{query}' without citing sources"
        )
        results = [SearchResult(title=query, url="", snippet=text)]

    return SearchResponse(results=results)
