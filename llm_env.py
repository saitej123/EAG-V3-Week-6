"""
LLM credentials and model list: read from the process environment.

On import, loads repository-root `.env` via `python-dotenv`, then exposes accessors.
Variable names are constants below; never embed secrets or model IDs in code.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Resolve repo-root `.env` whenever this module is imported (before any LLM reads).
_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")

# Environment variable names only — values come from `.env` / process env after load above.
_VAR_GEMINI_API_KEY = "GEMINI_API_KEY"
_VAR_GEMINI_MODELS = "GEMINI_MODELS"
_VAR_GEMINI_MODEL = "GEMINI_MODEL"
_VAR_TAVILY_API_KEY = "TAVILY_API_KEY"


def gemini_api_key() -> str:
    """Gemini API credential from the environment (empty if unset)."""
    return (os.environ.get(_VAR_GEMINI_API_KEY) or "").strip()


def tavily_api_key() -> str:
    """Tavily API credential from the environment (empty if unset)."""
    return (os.environ.get(_VAR_TAVILY_API_KEY) or "").strip()


def gemini_models_ordered() -> list[str]:
    """Models from env: `GEMINI_MODELS` (comma-separated) if set; else `GEMINI_MODEL` (single)."""
    raw = (os.environ.get(_VAR_GEMINI_MODELS) or "").strip()
    if raw:
        seen: set[str] = set()
        out: list[str] = []
        for part in raw.split(","):
            m = part.strip()
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out
    single = (os.environ.get(_VAR_GEMINI_MODEL) or "").strip()
    return [single] if single else []


# Lazy shared Gemini HTTP client (used by perception, decision, action — direct google-genai SDK).
_gemini_lock = threading.Lock()
_gemini_client_singleton: Any = False  # False = not yet resolved


def shared_gemini_client() -> Any:
    """Return a single cached ``google.genai.Client`` or ``None``; builds on first use only."""
    global _gemini_client_singleton
    if _gemini_client_singleton is not False:
        return _gemini_client_singleton
    with _gemini_lock:
        if _gemini_client_singleton is not False:
            return _gemini_client_singleton
        key = gemini_api_key()
        if not key:
            _gemini_client_singleton = None
            return None
        try:
            from google.genai import Client

            _gemini_client_singleton = Client(api_key=key)
        except Exception as e:
            from loguru import logger

            logger.warning(f"Shared Gemini client init failed: {e}")
            _gemini_client_singleton = None
        return _gemini_client_singleton


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def agent_max_iterations() -> int:
    """Max perceive→decide→act loops per agent run (default 5). Override with AGENT_MAX_ITERATIONS."""
    return max(1, min(50, _int_env("AGENT_MAX_ITERATIONS", 5)))


def agent_run_max_seconds() -> float:
    """Hard cap for one agent job (wall clock). Prevents Run agent staying busy forever."""
    return max(120.0, _float_env("AGENT_RUN_MAX_SECONDS", 900.0))


def agent_llm_step_timeout_seconds() -> float:
    """Perception / decision LLM call budget (each)."""
    return max(20.0, _float_env("AGENT_LLM_STEP_TIMEOUT_SEC", 120.0))


def mcp_tool_timeout_seconds(tool_name: str) -> float:
    """Per-tool MCP RPC budget; crawl-heavy tools get more time."""
    env_key = f"MCP_TIMEOUT_{tool_name.upper().replace('-', '_')}"
    if os.environ.get(env_key):
        return max(5.0, _float_env(env_key, 120.0))
    defaults: dict[str, float] = {
        "fetch_urls": 420.0,
        "fetch_url": 180.0,
        "web_search": 55.0,
        "query_database": 35.0,
        "analyze_image_url": 120.0,
        "gemini_live_search": 95.0,
        "get_time": 15.0,
        "currency_convert": 25.0,
        "read_file": 20.0,
        "list_dir": 15.0,
        "create_file": 20.0,
        "update_file": 20.0,
        "edit_file": 20.0,
    }
    return max(10.0, defaults.get(tool_name, 120.0))
