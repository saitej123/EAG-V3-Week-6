import asyncio
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from artifact_store import ARTIFACT_THRESHOLD_BYTES, ArtifactStore
from llm_env import gemini_api_key, gemini_models_ordered, mcp_tool_timeout_seconds, shared_gemini_client
from schemas import ToolCall

_PROJECT_ROOT = Path(__file__).resolve().parent
USAGE_PATH = _PROJECT_ROOT / "usage.json"


def _project_venv_python(project_root: Path) -> str | None:
    """Prefer repo `.venv` so MCP matches `uv sync` deps even if another venv is activated."""
    if platform.system() == "Windows":
        exe = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        exe = project_root / ".venv" / "bin" / "python"
    return str(exe) if exe.is_file() else None


def _mcp_python_executable() -> str:
    return _project_venv_python(_PROJECT_ROOT) or sys.executable


def _flatten_mcp_error(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        parts = [_flatten_mcp_error(e) for e in exc.exceptions]
        return " | ".join(p for p in parts if p)
    return f"{type(exc).__name__}: {exc}"


def _extract_mcp_tool_text(result: Any) -> str:
    """Normalize MCP CallToolResult content; avoids IndexError on empty or non-text blocks."""
    if result is None:
        return "(MCP returned no result.)"
    blocks = getattr(result, "content", None)
    if not blocks:
        return "(MCP tool finished with no content blocks.)"
    texts: list[str] = []
    for block in blocks:
        chunk = getattr(block, "text", None)
        if chunk is not None and str(chunk).strip():
            texts.append(str(chunk))
    if texts:
        return "\n".join(texts)
    return "(MCP returned content blocks without text; check server/tool implementation.)"


def _args_contain_art_prefix(obj: Any) -> bool:
    if isinstance(obj, str):
        return obj.strip().startswith("art:")
    if isinstance(obj, dict):
        return any(_args_contain_art_prefix(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_args_contain_art_prefix(v) for v in obj)
    return False


def _log_usage_snapshot(prefix: str = "[Tavily/DDG usage]") -> None:
    """Surface MCP search billing counters from usage.json for the UI stream."""
    if not USAGE_PATH.exists():
        logger.info(f"{prefix} usage.json not present yet")
        return
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        tv = data.get("tavily", {})
        dd = data.get("duckduckgo", {})
        logger.info(
            f"{prefix} month={data.get('month')} "
            f"tavily_calls={tv.get('count', 0)} tavily_errors={tv.get('errors', 0)} "
            f"ddg_calls={dd.get('count', 0)} ddg_errors={dd.get('errors', 0)}"
        )
    except Exception as e:
        logger.warning(f"{prefix} could not read usage.json: {e}")


class ActionActuator:
    def __init__(self):
        mcp_py = _mcp_python_executable()
        if mcp_py != sys.executable:
            logger.info(f"[MCP] Using project venv interpreter for subprocess: {mcp_py}")
        self.server_params = StdioServerParameters(
            command=mcp_py,
            args=["mcp_server.py"],
            env=os.environ.copy(),
            cwd=str(_PROJECT_ROOT),
        )

        self._mcp_lock = asyncio.Lock()
        self._stdio_cm: Any = None
        self._session_cm: Any = None
        self._mcp_session: ClientSession | None = None

    async def _reset_mcp_connection(self) -> None:
        """Close MCP stdio transport so the next tool call spawns a fresh server process."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self._mcp_session = None
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_cm = None

    async def _ensure_mcp_session(self) -> ClientSession:
        if self._mcp_session is not None:
            return self._mcp_session
        self._stdio_cm = stdio_client(self.server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._mcp_session = await self._session_cm.__aenter__()
        await self._mcp_session.initialize()
        logger.info("[MCP] Session initialized (stdio subprocess).")
        return self._mcp_session

    def _pack_descriptor(self, text: str, artifact_id: str | None) -> str:
        if artifact_id:
            prev = text[:400].replace("\n", " ")
            return f"[artifact {artifact_id}, {len(text.encode('utf-8'))} bytes] preview: {prev!r}"
        return text

    async def execute(self, tool_call: ToolCall, *, store: ArtifactStore) -> tuple[str, str | None]:
        """Session 6 dispatch: returns ``(descriptor_text, optional_artifact_id)``."""
        tool_name = (tool_call.name or "").strip()
        tool_args: dict[str, Any] = dict(tool_call.arguments)

        if _args_contain_art_prefix(tool_args):
            msg = (
                "Refused tool dispatch: arguments contain an internal `art:` handle. "
                "Use ATTACHED ARTIFACT bytes from context instead of passing handles to MCP paths."
            )
            logger.warning(f"[Action] {msg}")
            return msg, None

        if tool_name == "gemini_live_search":
            q = tool_args.get("query", "")
            budget = mcp_tool_timeout_seconds("gemini_live_search")
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(self.gemini_live_search, q),
                    timeout=budget,
                )
            except asyncio.TimeoutError:
                logger.error(f"[Gemini live search] timed out after {budget}s")
                text = (
                    f"Gemini live search timed out after {budget}s. "
                    "Use web_search or fetch_urls for faster discovery, then fetch PDPs."
                )
            raw = text.encode("utf-8")
            if len(raw) > ARTIFACT_THRESHOLD_BYTES:
                aid = store.put(raw, content_type="text/plain; charset=utf-8", source="gemini_live_search")
                return self._pack_descriptor(text, aid or None), aid or None
            return text, None

        logger.info(f"[MCP] --> {tool_name} args={tool_args!r}")

        tool_fail_msg: str | None = None
        conn_fail_msg: str | None = None
        text = ""
        last_conn_err: BaseException | None = None

        async with self._mcp_lock:
            for attempt in range(2):
                try:
                    session = await self._ensure_mcp_session()
                    try:
                        budget = mcp_tool_timeout_seconds(tool_name)
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, tool_args),
                            timeout=budget,
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[MCP] {tool_name} timed out after {budget}s — resetting session")
                        await self._reset_mcp_connection()
                        tool_fail_msg = (
                            f"Tool '{tool_name}' timed out after {budget}s "
                            "(page crawl or search may be stuck — retry with fewer URLs or a simpler query)."
                        )
                        break
                    except Exception as e:
                        tool_fail_msg = f"Tool execution failed: {e}"
                        logger.error(f"[MCP] <-- {tool_name} FAILED: {e}")
                        break
                    text = _extract_mcp_tool_text(result)
                    break
                except asyncio.CancelledError:
                    raise
                except BaseException as e:
                    last_conn_err = e
                    detail = _flatten_mcp_error(e)
                    logger.warning(
                        f"[MCP] transport/session error ({tool_name}), attempt {attempt + 1}/2: {detail}"
                    )
                    await self._reset_mcp_connection()
            else:
                detail = _flatten_mcp_error(last_conn_err) if last_conn_err else "unknown"
                logger.error(f"[MCP] subprocess/session failed ({tool_name}) after retries: {detail}")
                conn_fail_msg = (
                    f"MCP connection failed ({tool_name}): {detail}. "
                    "If imports failed in mcp_server.py, run `uv sync` and start the app with `uv run python app.py` "
                    "so the MCP child uses this project's `.venv`. Retry the step once the server stays up."
                )

        if conn_fail_msg:
            return conn_fail_msg, None
        if tool_fail_msg:
            return tool_fail_msg, None

        preview = text if len(text) <= 1200 else text[:1200] + "…"
        logger.info(f"[MCP] <-- {tool_name} result_chars={len(text)} preview={preview!r}")

        if tool_name == "web_search":
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    logger.info(f"[MCP web_search] hits={len(parsed)}")
                    for i, hit in enumerate(parsed[:5], 1):
                        logger.info(f"[MCP web_search] #{i} title={hit.get('title','')!r} url={hit.get('url','')!r}")
            except json.JSONDecodeError:
                logger.info("[MCP web_search] response was not JSON list")
            _log_usage_snapshot()

        if tool_name == "fetch_urls":
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    logger.info(f"[MCP fetch_urls] pages={len(parsed)}")
                    for i, pg in enumerate(parsed[:6], 1):
                        if isinstance(pg, dict):
                            logger.info(
                                f"[MCP fetch_urls] #{i} url={pg.get('url','')!r} "
                                f"status={pg.get('status')} chars={len(str(pg.get('text','')))}"
                            )
            except json.JSONDecodeError:
                logger.info("[MCP fetch_urls] response was not JSON list")

        raw = text.encode("utf-8")
        if len(raw) > ARTIFACT_THRESHOLD_BYTES:
            aid = store.put(
                raw,
                content_type="application/json" if tool_name in {"web_search", "fetch_urls", "fetch_url"} else "text/plain; charset=utf-8",
                source=f"mcp:{tool_name}",
                descriptor=f"{tool_name} result",
            )
            return self._pack_descriptor(text, aid or None), aid or None
        return text, None

    async def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Backward-compatible helper returning text only (used by tests / callers)."""
        desc, _aid = await self.execute(ToolCall(name=tool_name, arguments=tool_args), store=ArtifactStore())
        return desc

    def gemini_live_search(self, query: str) -> str:
        """Triggers Gemini's native Google Search tool for sanity-checking live prices."""
        from google.genai import types

        if not query or not str(query).strip():
            return "Gemini live search skipped: empty query."
        client = shared_gemini_client()
        if client is None:
            return (
                "Gemini live search unavailable (client failed to initialize). "
                "Configure `.env` for Gemini and retry, or use web_search / fetch_url."
            )
        if not gemini_api_key():
            return (
                "Gemini credentials are not configured in the environment; skipping grounded search. "
                "Use web_search or fetch_url instead."
            )

        logger.info(f"[Gemini google_search grounding] query={query!r}")
        models_to_try = gemini_models_ordered()
        if not models_to_try:
            return (
                "Gemini live search is disabled: set the comma-separated models list in `.env` "
                "(see `.env.example`)."
            )
        last_err: Exception | None = None
        for model_id in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=(
                        "You are assisting shoppers in India. Search for live listings related to:\n"
                        f"{query}\n\n"
                        "Prioritize Indian retailers and INR pricing (Amazon.in, Flipkart, official brand India pages). "
                        "Do not emphasize Amazon.com, Walmart, Best Buy, or Target unless the query explicitly asks for US stores. "
                        "Return concise bullets with price hints and platform names ONLY if search results support them. "
                        "If inconclusive, say so — do NOT invent SKUs, specs, or exact PDP facts."
                    ),
                    config=types.GenerateContentConfig(
                        tools=[{"google_search": {}}],
                        temperature=0.1,
                    ),
                )
                out = (response.text or "").strip()
                pv = out if len(out) <= 1500 else out[:1500] + "…"
                logger.info(
                    f"[Gemini google_search grounding] model={model_id} response_chars={len(out)} preview={pv!r}"
                )
                return out or "(Gemini returned an empty response.)"
            except Exception as e:
                last_err = e
                logger.warning(f"[Gemini google_search grounding] model={model_id} failed: {e}")
        logger.error(f"[Gemini google_search grounding] all models failed: {last_err!r}")
        return f"Gemini live search failed after fallbacks: {last_err or 'unknown error'}"
