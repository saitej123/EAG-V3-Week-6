"""
MCP server for EAGV3 Session 6.

Nine tools plus catalog query, stdio transport:
    web_search, fetch_url, fetch_urls (parallel PDP batch), analyze_image_url, query_database, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file

web_search:  Tavily primary, DuckDuckGo fallback. Hard-capped at 5 results.
fetch_url:   crawl4ai only — clean markdown via headless Chromium.
Usage for tavily and duckduckgo is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from typing import Any
os.environ["CRAWL4AI_BASE_DIRECTORY"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".crawl4ai"))

import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result
SEARCH_TIMEOUT_SEC = 18.0
# Avoid huge JSON-RPC payloads that can drop the MCP stdio connection.
MAX_FETCH_MARKDOWN_CHARS = 50_000
# Parallel batch fetch: up to 3 URLs, 3 concurrent browser workers.
MAX_FETCH_URLS_BATCH = 3
MAX_FETCH_URL_CONCURRENCY = 3
MAX_FETCH_MARKDOWN_CHARS_BATCH_URL = 40_000
CRAWLER_POOL_SIZE = 2
PAGE_TIMEOUT_MS = 25_000

_crawler_pool: asyncio.Queue[Any] | None = None
_pool_init_lock = asyncio.Lock()
_crawl_io_lock = asyncio.Lock()

load_dotenv(Path(__file__).parent / ".env")

from llm_env import gemini_api_key, gemini_models_ordered, tavily_api_key
from search_providers import async_ddg_html, httpx_plain_fetch, merge_search_hits, web_search_with_fallbacks

mcp = FastMCP("eagv3-s6-server")

# NOTE: Do not import google.genai at module scope. A broken/partial google-genai install
# would crash this process on startup and kill MCP stdio — breaking fetch_url/web_search.
# Gemini is lazy-imported only inside analyze_image_url.

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from search_providers import tavily_search as _tavily

    return _tavily(query, max_results)


def _ddg_search(query: str, max_results: int) -> list[dict]:
    from search_providers import ddg_search as _ddg

    return _ddg(query, max_results)


async def _async_tavily(query: str, max_results: int) -> list[dict]:
    from search_providers import async_tavily as _at

    if not tavily_api_key() or not _under_cap("tavily"):
        return []
    results = await _at(query, max_results)
    if results:
        _bump("tavily")
    elif tavily_api_key():
        _bump("tavily", "errors")
    return results


async def _async_ddg(query: str, max_results: int) -> list[dict]:
    from search_providers import async_ddg as _ad

    results = await _ad(query, max_results)
    if results:
        _bump("duckduckgo")
    return results


async def _init_crawler_pool() -> asyncio.Queue[Any] | None:
    global _crawler_pool
    async with _pool_init_lock:
        if _crawler_pool is not None:
            return _crawler_pool
        try:
            from crawl4ai import AsyncWebCrawler

            pool: asyncio.Queue[Any] = asyncio.Queue(maxsize=CRAWLER_POOL_SIZE)
            for _ in range(CRAWLER_POOL_SIZE):
                crawler = AsyncWebCrawler(verbose=False)
                await crawler.__aenter__()
                await pool.put(crawler)
            _crawler_pool = pool
            return pool
        except Exception:
            _crawler_pool = None
            return None


async def _borrow_crawler() -> Any | None:
    pool = await _init_crawler_pool()
    if pool is None:
        return None
    return await pool.get()


async def _return_crawler(crawler: Any | None) -> None:
    if crawler is not None and _crawler_pool is not None:
        await _crawler_pool.put(crawler)


def _httpx_fetch_fallback(url: str, cap: int, reason: str = "") -> dict:
    payload = httpx_plain_fetch(url, max_chars=cap)
    if reason:
        payload["crawl_fallback_reason"] = reason
    return payload


async def _crawl4ai_fetch(url: str, max_markdown_chars: int | None = None) -> dict:
    cap = max_markdown_chars if max_markdown_chars is not None else MAX_FETCH_MARKDOWN_CHARS
    crawler = await _borrow_crawler()
    if crawler is None:
        return _httpx_fetch_fallback(url, cap, reason="crawler_pool_unavailable")

    try:
        async with _crawl_io_lock:
            saved_fd = os.dup(1)
            os.dup2(2, 1)
            try:
                try:
                    from crawl4ai import CrawlerRunConfig

                    run_cfg = CrawlerRunConfig(page_timeout=PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                    r = await asyncio.wait_for(
                        crawler.arun(url=url, config=run_cfg),
                        timeout=PAGE_TIMEOUT_MS / 1000 + 10,
                    )
                except (ImportError, TypeError):
                    r = await asyncio.wait_for(
                        crawler.arun(url=url),
                        timeout=PAGE_TIMEOUT_MS / 1000 + 10,
                    )
            finally:
                os.dup2(saved_fd, 1)
                os.close(saved_fd)
        md = r.markdown
        raw = (
            getattr(md, "raw_markdown", None)
            or getattr(md, "fit_markdown", None)
            or md
            or r.cleaned_html
            or r.html
            or ""
        )
        text = str(raw).strip()
        if not text:
            return _httpx_fetch_fallback(url, cap, reason="empty_crawl_markdown")
        truncated = False
        if len(text) > cap:
            text = text[:cap]
            truncated = True
        payload: dict = {
            "status": int(getattr(r, "status_code", None) or 200),
            "content_type": "text/markdown",
            "length_bytes": len(text.encode("utf-8")),
            "text": text,
        }
        if truncated:
            payload["truncated"] = True
            payload["note"] = (
                f"Markdown truncated to {cap} chars for MCP transport; "
                "use a narrower fetch or search snippets if you need the tail."
            )
        return payload
    except asyncio.TimeoutError:
        return _httpx_fetch_fallback(url, cap, reason="crawl_timeout")
    except Exception as e:
        return _httpx_fetch_fallback(url, cap, reason=f"crawl_error:{type(e).__name__}")
    finally:
        await _return_crawler(crawler)


@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> str:
    """Search: Tavily + DDG parallel, merge, HTML DDG fallback. Returns JSON array string."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    q = (query or "").strip()
    if not q:
        return json.dumps([{"title": "web_search error", "url": "", "snippet": "Empty query."}])
    try:
        tavily_hits, ddg_hits = await asyncio.wait_for(
            asyncio.gather(_async_tavily(q, max_results), _async_ddg(q, max_results)),
            timeout=SEARCH_TIMEOUT_SEC,
        )
        merged = merge_search_hits(tavily_hits, ddg_hits, max_results=max_results)
        if merged:
            return json.dumps(merged, ensure_ascii=False)
        html_hits = await async_ddg_html(q, max_results)
        if html_hits:
            return json.dumps(html_hits, ensure_ascii=False)
        return json.dumps([
            {
                "title": "web_search error",
                "url": "",
                "snippet": "No results from Tavily, DuckDuckGo, or HTML fallback.",
            }
        ])
    except Exception:
        hits = await web_search_with_fallbacks(q, max_results)
        return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> str:
    """Fetch clean markdown from a URL via crawl4ai. Returns JSON object string."""
    return json.dumps(await _crawl4ai_fetch(url), ensure_ascii=False)


@mcp.tool()
async def fetch_urls(urls: list[str]) -> str:
    """Fetch up to 3 URLs in parallel via crawl4ai (warm browser pool). Returns JSON array string."""
    if not urls:
        return "[]"
    seen: set[str] = set()
    cleaned: list[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) >= MAX_FETCH_URLS_BATCH:
            break

    sem = asyncio.Semaphore(MAX_FETCH_URL_CONCURRENCY)

    async def _one(target: str) -> dict:
        async with sem:
            try:
                payload = await _crawl4ai_fetch(
                    target, max_markdown_chars=MAX_FETCH_MARKDOWN_CHARS_BATCH_URL
                )
                return {"url": target, **payload}
            except Exception as e:
                fb = _httpx_fetch_fallback(
                    target, MAX_FETCH_MARKDOWN_CHARS_BATCH_URL, reason=f"fetch_exception:{type(e).__name__}"
                )
                return {"url": target, **fb}

    raw_pages = await asyncio.gather(*[_one(u) for u in cleaned], return_exceptions=True)
    pages: list[dict] = []
    for i, item in enumerate(raw_pages):
        target = cleaned[i]
        if isinstance(item, Exception):
            fb = _httpx_fetch_fallback(
                target, MAX_FETCH_MARKDOWN_CHARS_BATCH_URL, reason=f"gather_exception:{type(item).__name__}"
            )
            pages.append({"url": target, **fb})
        elif isinstance(item, dict):
            pages.append(item)
    return json.dumps(pages, ensure_ascii=False)


@mcp.tool()
def query_database(search_term: str = "") -> list[dict]:
    """Search cached products in state/commerce.db (same DB as the agent MemoryManager). Empty term returns recent rows (limit 50)."""
    db_path = Path(__file__).resolve().parent / "state" / "commerce.db"
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        q = (
            "SELECT url, platform, product_name, base_price, net_price, bank_offers_text, scraped_at "
            "FROM products"
        )
        params: list = []
        st = search_term.strip()
        if st:
            q += " WHERE product_name LIKE ? OR url LIKE ? OR platform LIKE ?"
            pat = f"%{st}%"
            params.extend([pat, pat, pat])
        q += " ORDER BY scraped_at DESC LIMIT 50"
        cur.execute(q, params)
        return [dict(row) for row in cur.fetchall()]


@mcp.tool()
def analyze_image_url(url: str, prompt: str = "Describe this image in detail, extracting any product information, brand, prices, and text.") -> str:
    """Download an image from a URL and analyze its content using Gemini. Example: analyze_image_url("https://example.com/image.jpg")"""
    import httpx

    try:
        from google.genai import Client, types as genai_types
    except ImportError as e:
        return (
            "Gemini SDK failed to import inside MCP server. "
            "Repair your environment with: `uv pip install --force-reinstall 'google-genai>=2.3.0'` "
            f"(import error: {e})."
        )

    models = gemini_models_ordered()
    if not models:
        return "Image analysis needs LLM models configured in `.env` (comma-separated list; see `.env.example`)."

    try:
        client = Client(api_key=gemini_api_key())
        with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}) as hc:
            r = hc.get(url)
            r.raise_for_status()
            image_bytes = r.content
            mime_type = r.headers.get("content-type", "image/jpeg")

        last_err: Exception | None = None
        for model_id in models:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[
                        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                text = (response.text or "").strip()
                return text or "(Model returned an empty response.)"
            except Exception as e:
                last_err = e
        return f"Failed to analyze image from {url} after trying configured models: {last_err}"
    except Exception as e:
        return f"Failed to analyze image from {url}: {e}"



@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Example: get_time("Asia/Kolkata")."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox. Example: read_file("notes.txt")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    if not p.parent.exists():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
