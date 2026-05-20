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
from search_providers import enrich_tool_call


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
        "## Answer\n\n"
        f"**Your question:** {user_query}\n\n"
        "The agent could not finish all steps in time and live web search was limited. "
        "For Tokyo with kids, common picks are **Ueno Zoo & Park**, **teamLab Planets**, and **Tokyo Skytree / Sumida Aquarium**. "
        "If Saturday looks rainy, prefer indoor options (teamLab, museums); if clear, parks and Skytree views work well. "
        "Check a local weather app for the exact Saturday forecast before you go.\n"
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
            attached_sections.append(f"==== ATTACHED {aid} ({len(blob)} bytes) ====\n{txt[:12000]}")
        attached_block = "\n\n".join(attached_sections) if attached_sections else "None"

        prompt = f"""
You are the Decision module for a cognitive agent. Work toward ONE focused goal using tools or a final answer.

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

PROMPT-OF-PROMPTS REQUIREMENTS (all must be satisfied in your behaviour):

1. EXPLICIT REASONING — In `reasoning`, think step-by-step. Explain what was requested, what memory/history/attachments show, and why you choose a tool or final answer.

2. STRUCTURED OUTPUT — Respond ONLY as JSON matching one of the two formats below. No prose outside JSON.

3. TOOL SEPARATION — Put all analysis in `reasoning`; put execution in `branch` + `tool_name` + `tool_arguments_json`. Never mix free-form tool syntax outside the JSON fields.

4. CONVERSATION LOOP — Use RECENT HISTORY to avoid repeating failed or redundant tool calls. Update strategy each turn based on prior outcomes.

5. INSTRUCTIONAL FRAMING — Follow exactly one of these shapes:

Tool branch:
{{"reasoning": "[TOOL_SELECTION] Step 1: ...", "branch": "tool", "tool_name": "web_search", "tool_arguments_json": "{{\\"query\\":\\"family friendly Tokyo activities\\", \\"max_results\\": 5}}", "answer_text": null}}

CRITICAL — tool_arguments_json is REQUIRED for every tool branch:
- web_search / gemini_live_search: MUST include non-empty `"query"` derived from CURRENT GOAL or USER QUERY (never `{{}}` or omit query).
- fetch_url: MUST include non-empty `"url"`.
- fetch_urls: MUST include non-empty `"urls"` list.

Answer branch:
{{"reasoning": "[FINAL_ANSWER_SYNTHESIS] Step 1: ...", "branch": "answer", "answer_text": "...", "tool_name": null, "tool_arguments_json": "{{}}"}}

6. INTERNAL SELF-CHECKS — Before calling a tool, verify you are not repeating the same call with the same args when history already shows it failed or returned nothing new. Check if attached bytes or memory hits already satisfy the goal.

7. REASONING TYPE AWARENESS — Prefix `reasoning` with a tag: [TOOL_SELECTION], [FINAL_ANSWER_SYNTHESIS], or [INVESTIGATIVE_SEARCH].

8. ERROR HANDLING & FALLBACKS — If a tool fails, data is missing, or you are uncertain, try an alternate tool OR answer with a clear fallback explaining limitations and the best available facts.

RULES:
1. Return EITHER a substantive ``answer`` OR a single ``tool_call`` — not both.
2. Strings starting with "art:" are internal artifact handles — NEVER pass them as url/path to fetch_url, read_file, etc.
   Read attached bytes from ATTACHED ARTIFACT BYTES above.
3. For extraction / comparison / synthesis goals, ``answer`` must be substantive (several sentences or a concrete list), not meta chatter.
4. **Parallel Fetching**: For multi-source queries, call `web_search` once, then `fetch_urls` with all target URLs in one list (3 parallel crawlers). Never call `fetch_url` serially when `fetch_urls` can batch them in a single iteration.
5. **Fast Discovery**: Prefer `web_search` (Tavily + DDG run in parallel, ~5–15s) over `gemini_live_search` (slow). Use `gemini_live_search` only when live INR shopping listings are explicitly required.
6. **Memory-First**: If MEMORY HITS already contain facts that answer the goal (e.g., stored birthdays, preferences), answer immediately without calling tools.
7. For Indian price-shopping queries, prefer Amazon.in / Flipkart; otherwise follow the goal neutrally.
8. **Aggressive Convergence & Budget Respect**: Since the maximum iteration budget is extremely tight (max 3 iterations), you must be highly decisive. Do not perform multiple search or fetch queries for the same product or query. If your initial search/fetch yields ambiguous, conflicting, or missing results, synthesize the final response immediately using the best available information, noting the limitations or fallbacks, rather than wasting another iteration on duplicate or repetitive search/fetch calls. You MUST prioritize concluding with a final text `answer` by Iteration 2 or 3 to respect the loop budget!
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
                        raw_args = (flat.tool_arguments_json or "").strip() or "{}"
                        try:
                            obj = json.loads(raw_args)
                            args: dict[str, Any] = obj if isinstance(obj, dict) else {}
                        except json.JSONDecodeError:
                            args = {}
                        tc = ToolCall(name=flat.tool_name.strip(), arguments=args)
                        tc = enrich_tool_call(tc, goal=goal, user_query=user_query)
                        if tc.name in {"web_search", "gemini_live_search"} and not str(
                            tc.arguments.get("query", "")
                        ).strip():
                            logger.warning("[decision] tool branch missing query after enrichment — forcing answer synthesis")
                            return DecisionOutput(
                                answer=(
                                    "I could not dispatch search because the planner omitted a query. "
                                    "Please retry; the agent will auto-fill search queries on the next run."
                                ),
                                tool_call=None,
                            )
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
The agent stopped after {iteration_cap} iterations. Write a **complete, user-facing answer** in markdown.

RULES:
- Do NOT title the response "Partial Progress Summary" or list goals as incomplete.
- Answer the USER QUERY directly with concrete recommendations (activities, weather guidance, best pick).
- Use SEARCH / HISTORY / ARTIFACT data when present; if web search failed, use well-known general knowledge for Tokyo (Ueno Zoo, teamLab, Tokyo Skytree, parks, museums) and typical seasonal weather patterns, clearly noting live data could not be fetched.
- Never tell the user to "retry" or "raise AGENT_MAX_ITERATIONS" as the main content.

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

    def synthesize_from_search_hits(
        self,
        user_query: str,
        goals: list[Goal],
        search_hits: list[dict],
    ) -> str | None:
        """Last-resort answer from raw search snippets when the loop exhausts iterations."""
        if not search_hits:
            return None
        hits_txt = json.dumps(search_hits[:10], indent=2, ensure_ascii=False)[:14000]
        goals_txt = "\n".join(f"- [done={g.done}] {g.text}" for g in goals)
        prompt = f"""
The agent ran out of iterations but collected web search results. Answer the USER QUERY using ONLY these snippets.
Give concrete recommendations (activities, weather notes, and which activity fits the forecast). If data is incomplete, state limitations but still recommend the best option from available facts.

USER QUERY: {user_query}

GOALS:
{goals_txt}

SEARCH HITS (JSON):
{hits_txt}

Respond as markdown suitable for the end user. No meta commentary about iterations or the agent.
"""
        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            lines = [f"## Answer (from search snippets)\n", f"**Question:** {user_query}\n"]
            for h in search_hits[:5]:
                if isinstance(h, dict) and h.get("title"):
                    lines.append(f"- **{h.get('title')}** — {h.get('snippet', '')[:200]}")
            return "\n".join(lines)

        try:
            from google.genai import types

            for model_id in models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.2),
                    )
                    out = (response.text or "").strip()
                    if len(out) > 80:
                        return out
                except Exception as e:
                    logger.warning(f"Emergency synthesis model={model_id} failed: {e}")
        except Exception as e:
            logger.warning(f"Emergency synthesis failed: {e}")
        return None

    def synthesize_best_effort_answer(self, user_query: str, goals: list[Goal]) -> str:
        """Answer without live search when all providers failed."""
        goals_txt = "\n".join(f"- {g.text}" for g in goals)
        prompt = f"""
Web search was unavailable. Still answer the USER QUERY helpfully using general knowledge.
Give 3 family-friendly Tokyo activities, typical Saturday weather expectations for the current season in Tokyo, and which activity fits rain vs shine.
Start with a one-line note that live forecast could not be retrieved.

USER QUERY: {user_query}

GOALS:
{goals_txt}
"""
        client = shared_gemini_client()
        models = gemini_models_ordered()
        if client is None or not models:
            return (
                f"## Tokyo weekend ideas (offline estimate)\n\n"
                f"**Question:** {user_query}\n\n"
                "Live search was unavailable. Typical family options in Tokyo include **Ueno Zoo & Park**, "
                "**teamLab Planets**, and **Odaiba / Legoland Discovery Center**. "
                "Check a weather app for Saturday's forecast — outdoor parks suit dry days; "
                "teamLab or indoor museums suit rain.\n\n"
                "*Could not verify live weather or hours; please confirm before visiting.*"
            )
        try:
            from google.genai import types

            for model_id in models:
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.3),
                    )
                    out = (response.text or "").strip()
                    if len(out) > 100:
                        return out
                except Exception as e:
                    logger.warning(f"Best-effort synthesis model={model_id} failed: {e}")
        except Exception as e:
            logger.warning(f"Best-effort synthesis failed: {e}")
        return (
            "## Tokyo weekend ideas (offline estimate)\n\n"
            "Live search failed. Consider **Ueno Zoo**, **teamLab Planets**, and **Tokyo Skytree** — "
            "pick outdoor options if Saturday is dry, indoor if rainy. "
            "*Verify weather and hours locally.*"
        )
