"""
Session 6 agent loop — wires Memory → Perception → Decision → Action (MCP).

History entries are plain dicts mirroring typed boundaries (answer vs action events).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
os.environ["CRAWL4AI_BASE_DIRECTORY"] = str(_BASE_DIR / ".crawl4ai")

from dotenv import load_dotenv
from loguru import logger

from action import ActionActuator
from artifact_store import ArtifactStore
from decision import DecisionModule, fallback_iteration_budget_markdown
from llm_env import agent_llm_step_timeout_seconds, agent_max_iterations
from memory import MemoryManager
from perception import PerceptionModule
from schemas import DecisionOutput, Goal, Observation, ToolCall
from search_providers import (
    derive_search_queries,
    enrich_tool_call,
    merge_search_hits,
    primary_search_query,
    web_search_with_fallbacks,
)

# Cap artifact bytes sent to Decision LLM (large Wikipedia pages slow synthesis ~80s+).
MAX_DECISION_ATTACH_CHARS = 12_000


def _truncate_attachment_blob(blob: bytes) -> bytes:
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        return blob[:MAX_DECISION_ATTACH_CHARS]
    if len(text) <= MAX_DECISION_ATTACH_CHARS:
        return blob
    return text[:MAX_DECISION_ATTACH_CHARS].encode("utf-8")


def _log_final_answer(answer: str) -> None:
    text = answer or ""
    logger.success(f">>> FINAL ANSWER <<<\n{text}")
    try:
        logger.info("[UI_RESULT_JSON] " + json.dumps({"text": text}, ensure_ascii=False))
    except (TypeError, ValueError):
        logger.info("[UI_RESULT_JSON] " + json.dumps({"text": "(Answer could not be encoded.)"}))


def _final_text_from_history(history: list[dict]) -> str | None:
    answers = [h.get("text") for h in history if h.get("kind") == "answer"]
    answers = [a for a in answers if isinstance(a, str) and a.strip()]
    return answers[-1].strip() if answers else None


def _search_hits_from_history(history: list[dict]) -> list[dict]:
    hits: list[dict] = []
    for entry in history:
        if entry.get("kind") != "action" or entry.get("tool") != "web_search":
            continue
        desc = entry.get("result_descriptor") or ""
        if not desc.strip().startswith("["):
            try:
                parsed = json.loads(desc)
                if isinstance(parsed, list):
                    hits.extend(x for x in parsed if isinstance(x, dict) and x.get("url"))
            except json.JSONDecodeError:
                pass
    return hits


class CognitiveAgent:
    def __init__(self) -> None:
        self.memory = MemoryManager()
        self.perception = PerceptionModule()
        self.decision = DecisionModule()
        self.action = ActionActuator()
        self.artifacts = ArtifactStore()

    async def _emergency_rescue_answer(self, user_query: str, goals: list[Goal], history: list[dict]) -> str | None:
        """Run focused searches and synthesize when the loop exits without a user-facing answer."""
        goal_text = next((g.text for g in goals if not g.done), goals[0].text if goals else "")
        queries = derive_search_queries(user_query, goal_text, limit=2)
        if not queries:
            queries = [primary_search_query(user_query, goal_text)]
        logger.warning(f"[emergency] Running rescue searches: {queries!r}")

        batches = await asyncio.gather(
            *[web_search_with_fallbacks(q, 5) for q in queries],
            return_exceptions=True,
        )
        lists = [b for b in batches if isinstance(b, list)]
        merged = merge_search_hits(*lists, max_results=10) if lists else []
        merged = [h for h in merged if h.get("url") and "web_search error" not in str(h.get("title", "")).lower()]
        if not merged:
            prior = _search_hits_from_history(history)
            merged = prior

        if not merged:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self.decision.synthesize_best_effort_answer, user_query, goals),
                    timeout=agent_llm_step_timeout_seconds(),
                )
            except asyncio.TimeoutError:
                return self.decision.synthesize_best_effort_answer(user_query, goals)

        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    self.decision.synthesize_from_search_hits,
                    user_query,
                    goals,
                    merged,
                ),
                timeout=agent_llm_step_timeout_seconds(),
            )
            return answer
        except asyncio.TimeoutError:
            logger.error("[emergency] synthesis timed out — returning snippet list")
            lines = [f"## Recommendations (search-based)\n", f"**Your question:** {user_query}\n"]
            for h in merged[:6]:
                lines.append(f"- **{h.get('title', 'Result')}** — {h.get('snippet', '')[:220]}")
            lines.append(
                "\n*Weather and activity fit could not be fully synthesized in time; "
                "review snippets above or retry.*"
            )
            return "\n".join(lines)

    async def run(self, user_query: str, max_iterations: int | None = None) -> None:
        cap = agent_max_iterations() if max_iterations is None else max(1, min(50, max_iterations))
        run_id = uuid.uuid4().hex[:8]
        history: list[dict] = []
        prior_goals: list[Goal] = []

        try:
            await self._run_loop(user_query, cap, run_id, history, prior_goals)
        finally:
            try:
                await self.action.aclose()
            except Exception as e:
                logger.warning(f"[MCP] cleanup after run failed: {e}")

    async def _run_loop(
        self,
        user_query: str,
        cap: int,
        run_id: str,
        history: list[dict],
        prior_goals: list[Goal],
    ) -> None:
        logger.info("=" * 50)
        logger.info("--- cognitive loop ---")
        logger.info(f"Query: {user_query}")
        logger.info(f"run_id={run_id} | Max iterations: {cap}")
        logger.info("=" * 50)

        # Durable-memory classification on raw user text (assignment contract).
        await asyncio.to_thread(
            lambda: self.memory.remember(user_query, source="user_query", run_id=run_id),
        )

        consecutive_errors = 0
        max_consecutive = 4

        for i in range(cap):
            try:
                logger.info(f"[Iteration {i + 1}/{cap}]")

                hits = await asyncio.to_thread(
                    lambda: self.memory.read(user_query, history, top_k=10),
                )
                logger.info(f"[memory.read] {len(hits)} ranked hits")

                try:
                    obs = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.perception.observe,
                            user_query,
                            hits,
                            history,
                            prior_goals,
                            run_id,
                        ),
                        timeout=agent_llm_step_timeout_seconds(),
                    )
                except asyncio.TimeoutError:
                    logger.error("[perception] LLM step timed out — reusing prior goal plan")
                    history.append({"iter": i + 1, "kind": "error", "detail": "perception_llm_timeout"})
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive:
                        _log_final_answer(
                            "## Run stopped (errors)\n\nPerception timed out repeatedly. "
                            "See **Live console** and retry."
                        )
                        logger.info("[agent] RUN_COMPLETE reason=error_abort")
                        break
                    if prior_goals:
                        obs = Observation(goals=list(prior_goals))
                    elif self.perception.state.goals:
                        obs = Observation(goals=list(self.perception.state.goals))
                    else:
                        obs = Observation(
                            goals=[
                                Goal(
                                    id=f"g-{uuid.uuid4().hex[:8]}",
                                    text=(user_query or "Fulfill the user request.")[:800],
                                    done=False,
                                )
                            ]
                        )
                    prior_goals = list(obs.goals)
                    continue
                prior_goals = list(obs.goals)

                for g in obs.goals:
                    att = f" attach={g.attach_artifact_id}" if g.attach_artifact_id else ""
                    logger.info(f"  [done={g.done}] {g.text[:120]}{'…' if len(g.text) > 120 else ''}{att}")

                if obs.all_done():
                    logger.info("All goals done. Concluding...")
                    ft = _final_text_from_history(history)
                    if ft:
                        _log_final_answer(ft)
                    else:
                        _log_final_answer(
                            "All goals were marked complete, but no decision **answer** was recorded in "
                            "this run’s history. Open **Live console** for tool outputs and Perception "
                            "goal lines, or retry with a clearer intent."
                        )
                    logger.info("[agent] RUN_COMPLETE reason=all_goals_done")
                    break

                goal = obs.next_unfinished()
                if goal is None:
                    logger.warning("No unfinished goal in this iteration — stopping.")
                    _log_final_answer(
                        "The agent stopped: no unfinished goal was available (empty or inconsistent plan). "
                        "Check **Live console** and retry."
                    )
                    logger.info("[agent] RUN_COMPLETE reason=no_active_goal")
                    break

                synth_kw = ("synthes", "extract", "list", "compare", "decide", "recommend", "analyze", "present", "summar", "formulate")
                if any(k in goal.text.lower() for k in synth_kw):
                    art_hits = [h for h in hits if h.artifact_id]
                    if art_hits and not goal.attach_artifact_id:
                        goal.attach_artifact_id = art_hits[-1].artifact_id
                        logger.info(f"[attach] synthesis guard → {goal.attach_artifact_id}")

                attached: list[tuple[str, bytes]] = []
                if goal.attach_artifact_id and self.artifacts.exists(goal.attach_artifact_id):
                    blob = self.artifacts.get_bytes(goal.attach_artifact_id)
                    if blob:
                        trimmed = _truncate_attachment_blob(blob)
                        attached.append((goal.attach_artifact_id, trimmed))
                        logger.info(
                            f"[attach] bytes for {goal.attach_artifact_id} "
                            f"({len(blob)} → {len(trimmed)} bytes for decision)"
                        )

                db_rows = await asyncio.to_thread(self.memory.query_products, "")

                try:
                    decision_out = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.decision.next_step,
                            goal,
                            hits,
                            attached,
                            history,
                            user_query,
                            db_rows,
                        ),
                        timeout=agent_llm_step_timeout_seconds(),
                    )
                except asyncio.TimeoutError:
                    logger.error("[decision] LLM step timed out — running heuristic web_search")
                    history.append({"iter": i + 1, "kind": "error", "detail": "decision_llm_timeout"})
                    tc = enrich_tool_call(
                        ToolCall(name="web_search", arguments={}),
                        goal=goal,
                        user_query=user_query,
                    )
                    desc, art_id = await self.action.execute(
                        tc,
                        store=self.artifacts,
                        fallback_query=primary_search_query(user_query, goal.text),
                    )
                    await asyncio.to_thread(
                        lambda: self.memory.record_outcome(
                            tool_call=tc,
                            result_text=desc,
                            artifact_id=art_id,
                            run_id=run_id,
                            goal_id=goal.id,
                        ),
                    )
                    history.append(
                        {
                            "iter": i + 1,
                            "kind": "action",
                            "goal_id": goal.id,
                            "tool": tc.name,
                            "arguments": tc.arguments,
                            "result_descriptor": desc[:800],
                            "artifact_id": art_id,
                            "note": "heuristic_after_decision_timeout",
                        }
                    )
                    consecutive_errors = 0
                    continue

                ans, tc = decision_out.resolved()
                if ans is not None:
                    history.append(
                        {
                            "iter": i + 1,
                            "kind": "answer",
                            "goal_id": goal.id,
                            "text": ans,
                        }
                    )
                    logger.info("[decision] ANSWER recorded.")
                    consecutive_errors = 0

                    is_last_goal = (obs.next_unfinished() == goal and sum(1 for g in obs.goals if not g.done) == 1)
                    is_synthesis_answer = any(k in goal.text.lower() for k in synth_kw) and len(ans.strip()) > 120
                    if is_last_goal or (i == cap - 1) or is_synthesis_answer:
                        logger.info("Concluding immediately with final answer...")
                        _log_final_answer(ans)
                        logger.info("[agent] RUN_COMPLETE reason=all_goals_done")
                        break
                    continue

                if tc is None:
                    logger.warning("[decision] No answer and no tool_call — skipping.")
                    history.append({"iter": i + 1, "kind": "error", "detail": "empty_decision"})
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive:
                        _log_final_answer(
                            "## Run stopped (errors)\n\nToo many consecutive failures or empty "
                            "decisions. See **Live console** for details."
                        )
                        logger.info("[agent] RUN_COMPLETE reason=error_abort")
                        break
                    continue

                tc = enrich_tool_call(tc, goal=goal, user_query=user_query)
                logger.info(f"-> TOOL_CALL {tc.name} args={tc.arguments!r}")
                desc, art_id = await self.action.execute(
                    tc,
                    store=self.artifacts,
                    fallback_query=primary_search_query(user_query, goal.text),
                )

                await asyncio.to_thread(
                    lambda: self.memory.record_outcome(
                        tool_call=tc,
                        result_text=desc,
                        artifact_id=art_id,
                        run_id=run_id,
                        goal_id=goal.id,
                    ),
                )

                history.append(
                    {
                        "iter": i + 1,
                        "kind": "action",
                        "goal_id": goal.id,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result_descriptor": desc[:800],
                        "artifact_id": art_id,
                    }
                )
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.exception(f"[agent] iteration {i + 1} failed: {e}")
                history.append({"iter": i + 1, "kind": "iteration_error", "error": str(e)})
                if consecutive_errors >= max_consecutive:
                    _log_final_answer(
                        "## Run stopped (errors)\n\nRepeated iteration exceptions. See **Live console**."
                    )
                    logger.info("[agent] RUN_COMPLETE reason=error_abort")
                    break

        else:
            logger.warning("Reached max iterations without completion.")
            ft = _final_text_from_history(history)
            if ft:
                _log_final_answer(ft)
                logger.info("[agent] RUN_COMPLETE reason=answer_in_history")
                return

            obs_goals = list(self.perception.state.goals)
            rescue = await self._emergency_rescue_answer(user_query, obs_goals, history)
            if rescue:
                _log_final_answer(rescue)
                logger.info("[agent] RUN_COMPLETE reason=emergency_rescue")
                return

            hits = await asyncio.to_thread(
                lambda: self.memory.read(user_query, history, top_k=12),
            )
            tail = json.dumps(history[-16:], indent=2, default=str)
            attached_txt = ""
            for g in obs_goals:
                if g.attach_artifact_id and self.artifacts.exists(g.attach_artifact_id):
                    b = self.artifacts.get_bytes(g.attach_artifact_id)
                    if b:
                        attached_txt += b.decode("utf-8", errors="replace")[:20000]
            db_rows = await asyncio.to_thread(self.memory.query_products, "")
            try:
                summary_md = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.decision.summarize_partial_progress,
                        user_query,
                        obs_goals,
                        hits,
                        db_rows,
                        tail,
                        attached_txt,
                        iteration_cap=cap,
                    ),
                    timeout=agent_llm_step_timeout_seconds(),
                )
            except asyncio.TimeoutError:
                summary_md = fallback_iteration_budget_markdown(user_query, obs_goals, tail, cap)
            _log_final_answer(summary_md)
            logger.info("[agent] RUN_COMPLETE reason=max_iterations")


if __name__ == "__main__":
    load_dotenv(_BASE_DIR / ".env")
    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date."
    )
    asyncio.run(CognitiveAgent().run(q))
