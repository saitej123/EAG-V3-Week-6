"""
Session 6 Decision: ``next_step`` returns ``DecisionOutput`` (answer OR tool_call).

Structured JSON via Gemini ``response_schema`` only (no regex on model output).
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from pydantic import ValidationError

from llm_env import gemini_models_ordered, shared_gemini_client
from schemas import CachedProductRow, DecisionLLMFlat, DecisionOutput, Goal, MemoryItem, PartialSummaryMarkdown, ToolCall


TOOL_CATALOG = """
Available MCP tools (pick exactly one tool_call when external work is needed):
- web_search: {"query": str, "max_results": int}
- fetch_urls: {"urls": list[str]}
- fetch_url: {"url": str}
- query_database: {"search_term": str}
- gemini_live_search: {"query": str}
- analyze_image_url: {"url": str, "prompt": str}
- create_file: {"path": str, "content": str}
- update_file / edit_file / read_file / list_dir as exposed by the MCP server.
"""


def fallback_iteration_budget_markdown(
    user_query: str,
    goals: list[Goal],
    recent_history: str,
    iteration_cap: int,
) -> str:
    g_lines = "\n".join(f"- [done={g.done}] {g.id}: {g.text}" for g in goals) or "- (no goals)"
    tail = (recent_history or "").strip()
    if len(tail) > 12000:
        tail = tail[-12000:] + "\n\n… (trace truncated)"
    return (
        "## Partial analysis — iteration budget reached\n\n"
        f"The agent stopped after **{iteration_cap}** iteration(s).\n\n"
        "### Your question\n\n"
        f"{user_query}\n\n"
        "### Goals\n\n"
        f"{g_lines}\n\n"
        "### Recent trace\n\n"
        f"```text\n{tail or '(empty)'}\n```\n\n"
        "### Next steps\n\n"
        "- Narrow the query or raise **`AGENT_MAX_ITERATIONS`**.\n"
    )


class DecisionModule:
    def next_step(
        self,
        goal: Goal,
        hits: list[MemoryItem],
        attached: list[tuple[str, bytes]],
        history: list[dict[str, Any]],
        user_query: str,
        db_rows: list[CachedProductRow],
    ) -> DecisionOutput:
        hits_txt = json.dumps([h.model_dump(mode="json") for h in hits[:24]], indent=2, default=str)[:16000]
        db_txt = json.dumps([r.model_dump() for r in db_rows], indent=2, default=str)[:8000]
        hist_txt = json.dumps(history[-16:], indent=2, default=str)[:12000]

        attached_sections: list[str] = []
        for aid, blob in attached:
            try:
                txt = blob.decode("utf-8", errors="replace")
            except Exception:
                txt = str(blob[:2000])
            attached_sections.append(f"==== ATTACHED {aid} ({len(blob)} bytes) ====\n{txt[:28000]}")
        attached_block = "\n\n".join(attached_sections) if attached_sections else "None"

        prompt = f"""
You are the Decision module (Session 6). Work toward ONE focused goal using tools or a final answer.

USER QUERY (original): {user_query}

CURRENT GOAL (single focus):
id={goal.id!r} done={goal.done} text={goal.text!r}

MEMORY HITS (structured JSON):
{hits_txt}

DATABASE SNAPSHOT (commerce cache):
{db_txt}

RECENT HISTORY:
{hist_txt}

ATTACHED ARTIFACT BYTES (decoded as UTF-8 when possible):
{attached_block}

{TOOL_CATALOG}

RULES:
1. Return EITHER a substantive ``answer`` OR a single ``tool_call`` — not both.
2. Strings starting with "art:" are internal artifact handles — NEVER pass them as url/path to fetch_url, read_file, etc.
   Read attached bytes from ATTACHED ARTIFACT BYTES above.
3. For extraction / comparison / synthesis goals, ``answer`` must be substantive (several sentences or a concrete list), not meta chatter.
4. Prefer ``fetch_urls`` with every URL you already discovered when multiple pages must be read.
5. For Indian price-shopping queries, prefer Amazon.in / Flipkart; otherwise follow the goal neutrally.

When calling a tool, respond with JSON: {{"branch":"tool","tool_name":"<name>","tool_arguments":{{...}}}} .
When answering with plain text, respond with JSON: {{"branch":"answer","answer_text":"..."}} .
"""

        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            return DecisionOutput(
                answer="Decision unavailable: configure Gemini in `.env`.",
                tool_call=None,
            )

        last_exc: Exception | None = None
        try:
            from google.genai import types

            for model_id in models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=DecisionLLMFlat,
                            temperature=0.2,
                        ),
                    )
                    raw = (response.text or "").strip()
                    data = json.loads(raw)
                    flat = DecisionLLMFlat.model_validate(data)
                    if flat.branch == "tool" and flat.tool_name:
                        tc = ToolCall(name=flat.tool_name.strip(), arguments=dict(flat.tool_arguments or {}))
                        return DecisionOutput(answer=None, tool_call=tc)
                    return DecisionOutput(answer=(flat.answer_text or "").strip() or "(empty answer)", tool_call=None)
                except (json.JSONDecodeError, ValidationError, Exception) as e:
                    last_exc = e
                    logger.warning(f"Decision model={model_id} failed: {e}")
        except Exception as e:
            last_exc = e
            logger.warning(f"Decision failed: {e}")

        return DecisionOutput(
            answer=f"Decision failed ({type(last_exc).__name__ if last_exc else 'unknown'}).",
            tool_call=None,
        )

    def summarize_partial_progress(
        self,
        user_query: str,
        goals: list[Goal],
        hits: list[MemoryItem],
        db_rows: list[CachedProductRow],
        recent_history: str,
        artifact_content: str,
        *,
        iteration_cap: int,
    ) -> str:
        goals_txt = "\n".join(f"- [done={g.done}] {g.text}" for g in goals)
        hits_txt = json.dumps([h.model_dump(mode="json") for h in hits[:16]], indent=2, default=str)
        db_txt = json.dumps([r.model_dump() for r in db_rows], indent=2, default=str)

        prompt = f"""
The agent stopped after {iteration_cap} iterations. Summarise partial PROGRESS as markdown.
Ground claims ONLY in the data below.

USER QUERY: {user_query}

GOALS:
{goals_txt}

MEMORY HITS:
{hits_txt}

DB:
{db_txt}

HISTORY SNIPPET:
{recent_history[:12000]}

ARTIFACT EXCERPT:
{artifact_content[:24000]}
"""

        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            return fallback_iteration_budget_markdown(user_query, goals, recent_history, iteration_cap)

        try:
            from google.genai import types

            for model_id in models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=PartialSummaryMarkdown,
                            temperature=0.2,
                        ),
                    )
                    raw = (response.text or "").strip()
                    data = json.loads(raw)
                    out = PartialSummaryMarkdown.model_validate(data)
                    if out.markdown_answer.strip():
                        return out.markdown_answer.strip()
                except Exception as e:
                    logger.warning(f"Partial summary model={model_id} failed: {e}")
        except Exception as e:
            logger.warning(f"Partial summary failed: {e}")

        return fallback_iteration_budget_markdown(user_query, goals, recent_history, iteration_cap)
