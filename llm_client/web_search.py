"""SearXNG Private Gateway web search client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebSearchConfig:
    base_url: str
    api_token: str
    max_results: int = 5
    snippet_max_length: int = 240
    timeout_s: float = 30.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_token)

    @property
    def agent_endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/search/agent"


class WebSearchError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def web_search_config_from_env(env: Mapping[str, Any]) -> WebSearchConfig | None:
    base_url = str(env.get("SEARXNG_BASE_URL") or "").strip()
    api_token = str(env.get("SEARXNG_API_TOKEN") or "").strip()
    if not (base_url and api_token):
        return None

    try:
        max_results = int(env.get("SEARXNG_MAX_RESULTS") or 5)
    except Exception:
        max_results = 5

    try:
        snippet_max_length = int(env.get("SEARXNG_SNIPPET_MAX_LENGTH") or 240)
    except Exception:
        snippet_max_length = 240

    try:
        timeout_s = float(env.get("SEARXNG_TIMEOUT_S") or 30.0)
    except Exception:
        timeout_s = 30.0

    return WebSearchConfig(
        base_url=base_url,
        api_token=api_token,
        max_results=max(1, max_results),
        snippet_max_length=max(50, snippet_max_length),
        timeout_s=max(1.0, timeout_s),
    )


def search_web(
    config: WebSearchConfig,
    query: str,
    *,
    max_results: int | None = None,
    language: str | None = None,
    categories: str | None = None,
) -> dict[str, Any]:
    """Call SearXNG /search/agent and return the parsed JSON response."""

    params: dict[str, str] = {"q": query}
    params["max_results"] = str(max_results or config.max_results)
    params["snippet_max_length"] = str(config.snippet_max_length)
    if language:
        params["language"] = language
    if categories:
        params["categories"] = categories

    url = config.agent_endpoint + "?" + urlencode(params)
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {config.api_token}")
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=config.timeout_s) as resp:
            data = json.loads(resp.read().decode())
            return data
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        raise WebSearchError(
            f"HTTP {e.code}: {e.reason}",
            status_code=e.code,
            body=body,
        ) from e
    except URLError as e:
        raise WebSearchError(f"Connection error: {e.reason}") from e
    except json.JSONDecodeError as e:
        raise WebSearchError(f"Invalid JSON response: {e}") from e


def summarize_search_result(data: dict[str, Any]) -> dict[str, Any]:
    """Build a structured summary from the /search/agent response."""
    results = data.get("results") or []
    summary: dict[str, Any] = {
        "status": "success",
        "query": data.get("query", ""),
        "result_count": len(results),
        "results": [
            {
                "rank": r.get("rank", i + 1),
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            }
            for i, r in enumerate(results)
        ],
    }

    answer_box = data.get("answer_box")
    if answer_box:
        summary["answer_box"] = answer_box

    infobox = data.get("infobox")
    if infobox:
        summary["infobox"] = infobox

    suggestions = data.get("suggestions")
    if suggestions:
        summary["suggestions"] = suggestions

    return summary


def summarize_search_error(error: WebSearchError) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "error", "error": str(error)}
    if error.status_code:
        result["http_status"] = error.status_code
    return result
