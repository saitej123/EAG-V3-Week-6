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
from schemas import DecisionOutput, Goal, ToolCall


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


class CognitiveAgent:
    def __init__(self) -> None:
        self.memory = MemoryManager()
        self.perception = PerceptionModule()
        self.decision = DecisionModule()
        self.action = ActionActuator()
        self.artifacts = ArtifactStore()

    async def run(self, user_query: str, max_iterations: int | None = None) -> None:
        cap = agent_max_iterations() if max_iterations is None else max(1, min(50, max_iterations))
        run_id = uuid.uuid4().hex[:8]
        history: list[dict] = []
        prior_goals: list[Goal] = []

        logger.info("=" * 50)
        logger.info("--- Session 6 cognitive loop ---")
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

                obs = await asyncio.to_thread(
                    self.perception.observe,
                    user_query,
                    hits,
                    history,
                    prior_goals,
                    run_id,
                )
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

                synth_kw = ("synthes", "extract", "list", "compare", "decide")
                if any(k in goal.text.lower() for k in synth_kw):
                    art_hits = [h for h in hits if h.artifact_id]
                    if art_hits and not goal.attach_artifact_id:
                        goal.attach_artifact_id = art_hits[-1].artifact_id
                        logger.info(f"[attach] synthesis guard → {goal.attach_artifact_id}")

                attached: list[tuple[str, bytes]] = []
                if goal.attach_artifact_id and self.artifacts.exists(goal.attach_artifact_id):
                    blob = self.artifacts.get_bytes(goal.attach_artifact_id)
                    if blob:
                        attached.append((goal.attach_artifact_id, blob))
                        logger.info(f"[attach] bytes for {goal.attach_artifact_id} ({len(blob)} bytes)")

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
                    logger.error("[decision] LLM step timed out — placeholder answer")
                    decision_out = DecisionOutput(
                        answer="Planner timed out for this step; inspect logs and retry.",
                        tool_call=None,
                    )

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
                    logger.info("[decision] ANSWER recorded — Perception will reconcile goal completion next iter.")
                    consecutive_errors = 0
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

                logger.info(f"-> TOOL_CALL {tc.name} args={tc.arguments!r}")
                desc, art_id = await self.action.execute(tc, store=self.artifacts)

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
            hits = await asyncio.to_thread(
                lambda: self.memory.read(user_query, history, top_k=12),
            )
            obs_goals = list(self.perception.state.goals)
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
