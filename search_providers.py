"""
Shared web search, fetch fallbacks, and tool-argument enrichment.

Used by mcp_server (MCP tools), action.py (direct fallback when MCP fails),
and agent6/decision (auto-fill empty tool args from user query + goal).
Never raises — always returns structured dicts/lists the agent can consume.
"""

from __future__ import annotations

import asyncio
import json
import re
from html import unescape
from typing import Any
from urllib.parse import quote_plus

import httpx
from duckduckgo_search import DDGS

from llm_env import tavily_api_key
from schemas import Goal, ToolCall

SEARCH_TIMEOUT_SEC = 18.0
HTTP_FETCH_TIMEOUT_SEC = 20.0
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _norm_hit(title: str, url: str, snippet: str) -> dict[str, str]:
    return {
        "title": (title or "").strip(),
        "url": (url or "").strip(),
        "snippet": (snippet or "").strip(),
    }


def tavily_search(query: str, max_results: int) -> list[dict[str, str]]:
    key = tavily_api_key()
    if not key:
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )
        return [
            _norm_hit(r.get("title", ""), r.get("url", ""), r.get("content", ""))
            for r in resp.get("results", [])
            if r.get("url")
        ]
    except Exception:
        return []


def ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    hits: list[dict] = []
    try:
        with DDGS(timeout=15) as ddgs:
            for backend in ("auto", "html", "lite"):
                try:
                    hits = list(ddgs.text(query, max_results=max_results, backend=backend))
                except Exception:
                    hits = []
                if hits:
                    break
    except Exception:
        return []
    return [
        _norm_hit(h.get("title", ""), h.get("href", ""), h.get("body", ""))
        for h in hits
        if h.get("href")
    ]


def ddg_html_fallback(query: str, max_results: int) -> list[dict[str, str]]:
    """Last-resort HTML scrape when DDGS library returns nothing."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        with httpx.Client(
            timeout=HTTP_FETCH_TIMEOUT_SEC,
            follow_redirects=True,
            headers=_HTTP_HEADERS,
        ) as client:
            r = client.get(url)
            r.raise_for_status()
            html = r.text
    except Exception:
        return []

    out: list[dict[str, str]] = []
    for block in re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        link, title_raw, snippet_raw = block
        title = unescape(re.sub(r"<[^>]+>", "", title_raw)).strip()
        snippet = unescape(re.sub(r"<[^>]+>", "", snippet_raw)).strip()
        if link and title:
            out.append(_norm_hit(title, link, snippet))
        if len(out) >= max_results:
            break
    return out


def merge_search_hits(
    *sources: list[dict[str, str]],
    max_results: int,
) -> list[dict[str, str]]:
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for src in sources:
        for hit in src:
            url = hit.get("url", "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            merged.append(hit)
            if len(merged) >= max_results:
                return merged
    return merged


async def async_tavily(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(tavily_search, query, max_results),
            timeout=SEARCH_TIMEOUT_SEC,
        )
    except Exception:
        return []


async def async_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(ddg_search, query, max_results),
            timeout=SEARCH_TIMEOUT_SEC,
        )
    except Exception:
        return []


async def async_ddg_html(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(ddg_html_fallback, query, max_results),
            timeout=SEARCH_TIMEOUT_SEC,
        )
    except Exception:
        return []


async def web_search_with_fallbacks(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Tavily + DDG in parallel, merge/dedupe, then HTML DDG fallback.
    Never raises.
    """
    q = (query or "").strip()
    if not q:
        return [_norm_hit("web_search error", "", "Empty query.")]

    max_results = max(1, min(max_results, 5))
    try:
        tavily_hits, ddg_hits = await asyncio.gather(
            async_tavily(q, max_results),
            async_ddg(q, max_results),
        )
        merged = merge_search_hits(tavily_hits, ddg_hits, max_results=max_results)
        if merged:
            return merged

        html_hits = await async_ddg_html(q, max_results)
        if html_hits:
            return html_hits

        return [
            _norm_hit(
                "web_search error",
                "",
                "No results from Tavily, DuckDuckGo, or HTML fallback. Check network/API keys.",
            )
        ]
    except Exception as e:
        return [_norm_hit("web_search error", "", f"{type(e).__name__}: {e}")]


def web_search_json(query: str, max_results: int) -> str:
    """Sync wrapper for MCP tools running in thread pool if needed."""
    return json.dumps(
        asyncio.run(web_search_with_fallbacks(query, max_results)),
        ensure_ascii=False,
    )


def httpx_plain_fetch(url: str, max_chars: int = 12_000) -> dict[str, Any]:
    """Lightweight HTTP fallback when crawl4ai is unavailable or times out."""
    target = (url or "").strip()
    if not target:
        return {
            "status": 0,
            "content_type": "text/plain",
            "length_bytes": 0,
            "text": "[httpx_fetch] Empty URL.",
            "error": "empty_url",
            "fallback": "httpx",
        }
    try:
        with httpx.Client(
            timeout=HTTP_FETCH_TIMEOUT_SEC,
            follow_redirects=True,
            headers=_HTTP_HEADERS,
        ) as client:
            r = client.get(target)
            r.raise_for_status()
            raw = r.text or ""
            text = unescape(re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.I | re.S))
            text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars]
            return {
                "status": r.status_code,
                "content_type": "text/plain",
                "length_bytes": len(text.encode("utf-8")),
                "text": text or "(empty body)",
                "fallback": "httpx",
            }
    except Exception as e:
        return {
            "status": 0,
            "content_type": "text/plain",
            "length_bytes": 0,
            "text": f"[httpx_fetch] Failed for {target!r}: {type(e).__name__}: {e}",
            "error": str(e),
            "fallback": "httpx",
        }


def is_search_error_payload(text: str) -> bool:
    """True when web_search JSON indicates failure (triggers action-layer retry)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return True
    items = data if isinstance(data, list) else [data]
    if not items:
        return True
    first = items[0] if isinstance(items[0], dict) else {}
    title = str(first.get("title", "")).lower()
    return title == "web_search error" or not first.get("url")


# --- Tool argument enrichment (was tool_enrichment.py) ------------------------


def derive_search_queries(user_query: str, goal_text: str = "", *, limit: int = 3) -> list[str]:
    """Build focused search strings from the user question and active goal."""
    uq = (user_query or "").strip()
    gt = (goal_text or "").strip()
    candidates: list[str] = []

    if uq:
        candidates.append(uq[:280])
    if gt and gt.lower() != uq.lower() and len(gt) > 12:
        candidates.append(gt[:280])

    lower = uq.lower()
    if "weather" in lower:
        city = _extract_place(uq) or "Tokyo"
        candidates.append(f"{city} weather forecast Saturday this weekend")
    if any(w in lower for w in ("family", "family-friendly", "kids", "children")):
        place = _extract_place(uq) or ""
        if place:
            candidates.append(f"family friendly things to do {place} weekend")

    seen: set[str] = set()
    out: list[str] = []
    for q in candidates:
        key = q.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= limit:
            break
    return out or ([uq[:280]] if uq else [])


def _extract_place(text: str) -> str:
    m = re.search(r"\bin\s+([A-Z][a-zA-Z\s\-]+?)(?:\s+this|\s+weekend|[,.]|$)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(Tokyo|Delhi|Mumbai|Bangalore|London|Paris|New York|Sydney)\b", text, re.I)
    return m.group(1) if m else ""


def enrich_tool_call(tc: ToolCall, *, goal: Goal, user_query: str) -> ToolCall:
    """Ensure required tool fields are populated before MCP dispatch."""
    name = (tc.name or "").strip()
    args = dict(tc.arguments or {})
    queries = derive_search_queries(user_query, goal.text)

    if name in {"web_search", "gemini_live_search"}:
        q = str(args.get("query") or args.get("q") or "").strip()
        if not q:
            q = queries[0] if queries else (goal.text.strip()[:280] or user_query.strip()[:280])
            args["query"] = q
        if name == "web_search":
            try:
                mr = int(args.get("max_results", 5))
            except (TypeError, ValueError):
                mr = 5
            args["max_results"] = max(1, min(mr, 5))

    elif name == "fetch_url":
        url = str(args.get("url") or "").strip()
        if not url and args.get("query"):
            args["url"] = str(args["query"]).strip()

    elif name == "fetch_urls":
        urls = args.get("urls")
        if not isinstance(urls, list) or not any(str(u).strip() for u in urls):
            args["urls"] = []

    return ToolCall(name=name, arguments=args)


def primary_search_query(user_query: str, goal_text: str = "") -> str:
    qs = derive_search_queries(user_query, goal_text, limit=1)
    return qs[0] if qs else (user_query or goal_text or "").strip()[:280]
