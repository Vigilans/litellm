from typing import Any, Dict, List, Tuple

from litellm._uuid import uuid
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.utils import (
    build_web_search_results_from_annotations as _build_web_search_results_from_annotations,
    build_text_blocks_with_citations,
)


def build_web_tool_use(web_search_query: str) -> Dict[str, Any]:
    return {
        "type": "server_tool_use",
        "id": f"srvtoolu_{uuid.uuid4().hex[:24]}",
        "name": "web_search",
        "input": {"query": web_search_query},
    }


def build_web_search_results_from_annotations(
    web_tool_uses: List[Dict[str, Any]],
    annotations: list,
) -> Tuple[List[Dict[str, Any]], List[tuple]]:
    flat: List[Dict[str, Any]] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            ann = ann.model_dump() if hasattr(ann, "model_dump") else dict(ann)
        inner = ann.get("url_citation", ann)
        flat.append({
            "type": "url_citation",
            "url": inner.get("url", ""),
            "title": inner.get("title", ""),
            "start_index": inner.get("start_index", 0) or 0,
            "end_index": inner.get("end_index", 0) or 0,
        })
    return _build_web_search_results_from_annotations(web_tool_uses, flat)
